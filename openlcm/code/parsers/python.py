"""Python AST → LST symbol/edge extractor.

Uses only the stdlib `ast` module — zero external dependencies.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Symbol:
    kind: str               # 'class' | 'function' | 'method' | 'import' | 'variable'
    name: str
    qualified_name: str
    parent_name: str = ""   # dotted qualified name of the enclosing class (for methods)
    signature: str = ""
    docstring: str = ""
    line_start: int = 0
    line_end: int = 0
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)


@dataclass
class Edge:
    from_qname: str         # qualified name of the calling/importing symbol
    edge_type: str          # 'calls' | 'imports' | 'inherits'
    to_name: str            # raw name referenced (may be unresolved)


class PythonASTParser:
    """Extract symbols and edges from Python source via stdlib ast."""

    def parse(self, source: str, file_path: str) -> tuple[list[Symbol], list[Edge]]:
        module_name = _module_name(file_path)
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return [], []

        symbols: list[Symbol] = []
        edges: list[Edge] = []

        self._visit_module(tree, module_name, symbols, edges)
        return symbols, edges

    # ── Internal traversal ────────────────────────────────────────────────

    def _visit_module(
        self,
        tree: ast.Module,
        module_name: str,
        symbols: list[Symbol],
        edges: list[Edge],
    ) -> None:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._visit_class(node, module_name, symbols, edges)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._visit_function(node, module_name, None, symbols, edges)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._visit_import(node, module_name, edges)
            elif isinstance(node, ast.Assign):
                self._visit_module_assign(node, module_name, symbols)

    def _visit_class(
        self,
        node: ast.ClassDef,
        module_name: str,
        symbols: list[Symbol],
        edges: list[Edge],
    ) -> None:
        qname = f"{module_name}.{node.name}" if module_name else node.name
        docstring = ast.get_docstring(node) or ""
        decorators = _decorator_names(node)
        bases = [_name_of(b) for b in node.bases if _name_of(b)]

        symbols.append(Symbol(
            kind="class",
            name=node.name,
            qualified_name=qname,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            decorators=decorators,
        ))

        for base in bases:
            edges.append(Edge(from_qname=qname, edge_type="inherits", to_name=base))

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._visit_function(child, module_name, qname, symbols, edges)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module_name: str,
        class_qname: Optional[str],
        symbols: list[Symbol],
        edges: list[Edge],
    ) -> None:
        if class_qname:
            qname = f"{class_qname}.{node.name}"
            kind = "method"
        else:
            qname = f"{module_name}.{node.name}" if module_name else node.name
            kind = "function"

        docstring = ast.get_docstring(node) or ""
        signature = _build_signature(node)
        decorators = _decorator_names(node)
        is_async = isinstance(node, ast.AsyncFunctionDef)

        symbols.append(Symbol(
            kind=kind,
            name=node.name,
            qualified_name=qname,
            parent_name=class_qname or "",
            signature=signature,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            is_async=is_async,
            decorators=decorators,
        ))

        # Extract call edges from the function body
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = _name_of(child.func)
                if callee:
                    edges.append(Edge(from_qname=qname, edge_type="calls", to_name=callee))

    def _visit_import(
        self,
        node: ast.Import | ast.ImportFrom,
        module_name: str,
        edges: list[Edge],
    ) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(Edge(
                    from_qname=module_name,
                    edge_type="imports",
                    to_name=alias.name,
                ))
        else:
            base = node.module or ""
            for alias in node.names:
                full = f"{base}.{alias.name}" if base else alias.name
                edges.append(Edge(
                    from_qname=module_name,
                    edge_type="imports",
                    to_name=full,
                ))

    def _visit_module_assign(
        self,
        node: ast.Assign,
        module_name: str,
        symbols: list[Symbol],
    ) -> None:
        for target in node.targets:
            name = _name_of(target)
            if not name or name.startswith("_"):
                continue
            value_repr = _simple_value(node.value)
            symbols.append(Symbol(
                kind="variable",
                name=name,
                qualified_name=f"{module_name}.{name}" if module_name else name,
                signature=f"= {value_repr}" if value_repr else "",
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
            ))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _module_name(file_path: str) -> str:
    """Convert a file path to a dotted module name (best effort)."""
    p = Path(file_path)
    # Strip __init__.py → parent package name
    if p.name == "__init__.py":
        parts = list(p.parent.parts)
    else:
        parts = list(p.parent.parts) + [p.stem]
    # Trim leading path until we hit a likely package root
    # (first component that doesn't contain src/site-packages/etc.)
    skip = {"src", "lib", "site-packages", "dist-packages", ""}
    trimmed = []
    for part in reversed(parts):
        if part in skip:
            break
        trimmed.insert(0, part)
    return ".".join(trimmed) if trimmed else p.stem


def _name_of(node: ast.expr) -> str:
    """Extract a dotted name from an AST expression (best effort)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name_of(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _decorator_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    names = []
    for d in node.decorator_list:
        n = _name_of(d)
        if n:
            names.append(n)
    return names


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a human-readable signature string from the AST."""
    args = node.args
    params: list[str] = []

    # positional args (with defaults right-aligned)
    defaults = args.defaults
    n_args = len(args.args)
    n_defaults = len(defaults)

    for i, arg in enumerate(args.args):
        default_idx = i - (n_args - n_defaults)
        part = arg.arg
        if arg.annotation:
            ann = _unparse_annotation(arg.annotation)
            part = f"{part}: {ann}"
        if default_idx >= 0:
            dflt = _unparse_annotation(defaults[default_idx])
            part = f"{part} = {dflt}"
        params.append(part)

    if args.vararg:
        part = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            part = f"*{args.vararg.arg}: {_unparse_annotation(args.vararg.annotation)}"
        params.append(part)
    elif args.kwonlyargs:
        params.append("*")

    for i, arg in enumerate(args.kwonlyargs):
        part = arg.arg
        if arg.annotation:
            part = f"{part}: {_unparse_annotation(arg.annotation)}"
        kw_defaults = args.kw_defaults
        if i < len(kw_defaults) and kw_defaults[i] is not None:
            part = f"{part} = {_unparse_annotation(kw_defaults[i])}"
        params.append(part)

    if args.kwarg:
        part = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            part = f"**{args.kwarg.arg}: {_unparse_annotation(args.kwarg.annotation)}"
        params.append(part)

    ret = ""
    if node.returns:
        ret = f" -> {_unparse_annotation(node.returns)}"

    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}({', '.join(params)}){ret}"


def _unparse_annotation(node: ast.expr) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return _name_of(node) or "..."


def _simple_value(node: ast.expr) -> str:
    """Return a short string representation of a simple constant/name value."""
    try:
        return ast.unparse(node)[:80]
    except Exception:
        pass
    if isinstance(node, ast.Constant):
        return repr(node.value)[:80]
    if isinstance(node, ast.Name):
        return node.id
    return ""
