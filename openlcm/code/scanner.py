"""RepoScanner — scans local paths or remote GitHub/GitLab URLs into an LSTGraph.

Pipeline:
  1. If input is a URL → git clone (or pull) into ~/.openlcm/repos/
  2. Walk repo: Python files → PythonASTParser (stdlib, docstrings + signatures + call edges)
  3. Non-Python files → CtagsParser (Universal Ctags subprocess, 100+ languages)
  4. Store git metadata (branch, commit hash, origin URL) in lcm_lst_repos table
  5. Incremental: skip files whose SHA-256 hash hasn't changed
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from .git import GitCloner, get_repo_metadata, is_url
from .graph import LSTGraph
from .parsers.python import PythonASTParser, Symbol, Edge

logger = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", "env", ".env", "dist", "build",
    ".tox", ".eggs", ".next", ".nuxt", "target", "vendor",
    "coverage", ".coverage", "htmlcov",
}

_LANGUAGE = {
    ".py":    "python",
    ".ts":    "typescript",
    ".tsx":   "typescript",
    ".js":    "javascript",
    ".jsx":   "javascript",
    ".mjs":   "javascript",
    ".go":    "go",
    ".java":  "java",
    ".rs":    "rust",
    ".rb":    "ruby",
    ".php":   "php",
    ".swift": "swift",
    ".kt":    "kotlin",
    ".kts":   "kotlin",
    ".c":     "c",
    ".h":     "c",
    ".cpp":   "cpp",
    ".cc":    "cpp",
    ".hpp":   "cpp",
    ".cs":    "csharp",
    ".scala": "scala",
    ".sh":    "shell",
    ".bash":  "shell",
}

# Extensions handled by Python AST (rich: docstrings, call edges, full sigs)
_PYTHON_EXTS = {".py"}
# Extensions handled by ctags (structural: classes, functions, imports)
_CTAGS_EXTS = set(_LANGUAGE.keys()) - _PYTHON_EXTS


class RepoScanner:
    """Scan a repository (local path or remote URL) into an LSTGraph."""

    def __init__(self, cache_dir: Optional[str | Path] = None) -> None:
        self._cloner = GitCloner(cache_dir=cache_dir)

    def scan(
        self,
        repo_path_or_url: str | Path,
        graph: LSTGraph,
        repo_id: str = "default",
        force: bool = False,
        branch: Optional[str] = None,
        clone_depth: int = 1,
        on_progress: Optional[Callable[[str, dict], None]] = None,
    ) -> dict:
        """
        Scan a repository and populate the LSTGraph.

        Accepts:
          - Local directory path: "/path/to/myapp"
          - GitHub/GitLab HTTPS URL: "https://github.com/user/repo"
          - SSH URL: "git@github.com:user/repo.git"

        Returns stats dict: scanned, skipped, errors, files, symbols, edges,
                            elapsed_s, languages (breakdown), git (metadata)
        """
        def _emit(event: str, data: dict = {}) -> None:
            if on_progress:
                try:
                    on_progress(event, data)
                except Exception:
                    pass

        t0 = time.time()
        clone_result = None

        path_str = str(repo_path_or_url).strip()
        if is_url(path_str):
            logger.info("Detected remote URL — cloning %s", path_str)
            _emit("clone_start", {"url": path_str})
            clone_result = self._cloner.clone(
                path_str,
                branch=branch,
                depth=clone_depth,
                force_refresh=force,
            )
            root = clone_result.local_path
            _emit("clone_done", {
                "url": path_str,
                "local_path": str(root),
                "was_cached": clone_result.was_cached,
            })
            # Use origin URL as default repo_id if caller left it as "default"
            if repo_id == "default":
                repo_id = _url_to_repo_id(path_str)
        else:
            root = Path(path_str).resolve()

        if not root.exists():
            return {"error": f"path does not exist: {root}"}

        # ── Git metadata ─────────────────────────────────────────────────
        if clone_result:
            git_meta = {
                "origin_url": clone_result.origin_url,
                "branch": clone_result.branch,
                "commit_hash": clone_result.commit_hash,
                "commit_message": clone_result.commit_message,
            }
        else:
            git_meta = get_repo_metadata(root)

        # Store repo record
        graph.upsert_repo(
            repo_id=repo_id,
            origin_url=git_meta.get("origin_url", ""),
            branch=git_meta.get("branch", ""),
            commit_hash=git_meta.get("commit_hash", ""),
            local_path=str(root),
        )

        # ── Python AST pass ──────────────────────────────────────────────
        scanned = skipped = errors = 0
        lang_counts: dict[str, int] = {}

        _emit("phase_start", {"phase": "python", "root": str(root)})

        for file_path in _walk(root):
            suffix = file_path.suffix.lower()
            if suffix not in _PYTHON_EXTS:
                continue

            rel_path = file_path.relative_to(root).as_posix()
            try:
                file_hash = _sha256(file_path)
            except OSError:
                errors += 1
                _emit("file_error", {"path": rel_path})
                continue

            if not force and graph.get_file_hash(repo_id, rel_path) == file_hash:
                skipped += 1
                _emit("file_skip", {"path": rel_path, "lang": "python"})
                continue

            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                errors += 1
                _emit("file_error", {"path": rel_path})
                continue

            lang = _LANGUAGE.get(suffix, "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            _emit("file_scan", {"path": rel_path, "lang": lang})

            file_id = graph.upsert_file(
                repo_id, rel_path, lang, file_hash, source.count("\n") + 1
            )
            graph.delete_file_symbols(file_id)

            symbols, edges = PythonASTParser().parse(source, rel_path)
            _ingest(graph, file_id, symbols, edges)
            scanned += 1

            _emit("file_done", {"path": rel_path, "lang": lang,
                                "symbols": len(symbols), "edges": len(edges)})

            if scanned % 100 == 0:
                logger.debug("LST scan: %d Python files", scanned)

        # ── Ctags pass (non-Python) ──────────────────────────────────────
        _emit("phase_start", {"phase": "ctags"})
        ctags_symbols = _run_ctags(root)
        ctags_by_file: dict[str, list] = {}
        for sym in ctags_symbols:
            ctags_by_file.setdefault(sym.file_path, []).append(sym)

        if ctags_by_file:
            _emit("ctags_files", {"count": len(ctags_by_file)})

        for rel_path, syms in ctags_by_file.items():
            file_path = root / rel_path
            suffix = file_path.suffix.lower()
            if suffix not in _CTAGS_EXTS:
                continue

            try:
                file_hash = _sha256(file_path)
            except OSError:
                errors += 1
                _emit("file_error", {"path": rel_path})
                continue

            if not force and graph.get_file_hash(repo_id, rel_path) == file_hash:
                skipped += 1
                _emit("file_skip", {"path": rel_path, "lang": syms[0].language if syms else "?"})
                continue

            lang = _LANGUAGE.get(suffix, syms[0].language if syms else "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            _emit("file_scan", {"path": rel_path, "lang": lang})

            try:
                line_count = file_path.read_bytes().count(b"\n") + 1
            except OSError:
                line_count = 0

            file_id = graph.upsert_file(repo_id, rel_path, lang, file_hash, line_count)
            graph.delete_file_symbols(file_id)

            _ingest_ctags(graph, file_id, syms)
            scanned += 1

            _emit("file_done", {"path": rel_path, "lang": lang, "symbols": len(syms), "edges": 0})

        stats = graph.get_stats(repo_id)
        return {
            "scanned": scanned,
            "skipped": skipped,
            "errors": errors,
            "files": stats["files"],
            "symbols": stats["symbols"],
            "edges": stats["edges"],
            "languages": lang_counts,
            "git": git_meta,
            "local_path": str(root),
            "elapsed_s": round(time.time() - t0, 2),
        }


# ── Internal ──────────────────────────────────────────────────────────────────

def _walk(root: Path):
    """Yield all files under root, skipping ignored directories."""
    try:
        children = sorted(root.iterdir())
    except PermissionError:
        return
    for child in children:
        if child.is_symlink():
            continue
        if child.is_dir():
            if child.name in _SKIP_DIRS or child.name.endswith(".egg-info"):
                continue
            yield from _walk(child)
        elif child.is_file():
            yield child


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ingest(
    graph: LSTGraph,
    file_id: int,
    symbols: list[Symbol],
    edges: list[Edge],
) -> None:
    """Insert Python AST symbols + edges into the graph."""
    qname_to_id: dict[str, int] = {}

    for sym in symbols:
        parent_id: Optional[int] = qname_to_id.get(sym.parent_name) if sym.parent_name else None
        sid = graph.insert_symbol(
            file_id, sym.kind, sym.name, sym.qualified_name,
            parent_symbol_id=parent_id,
            signature=sym.signature,
            docstring=sym.docstring,
            line_start=sym.line_start,
            line_end=sym.line_end,
            is_async=sym.is_async,
            decorators=sym.decorators,
        )
        qname_to_id[sym.qualified_name] = sid

    for edge in edges:
        graph.insert_edge(
            file_id, edge.edge_type, edge.to_name,
            from_symbol_id=qname_to_id.get(edge.from_qname),
        )


def _ingest_ctags(graph: LSTGraph, file_id: int, symbols: list) -> None:
    """Insert ctags symbols into the graph (no call edges — ctags doesn't provide them)."""
    qname_to_id: dict[str, int] = {}

    # First pass: classes before methods so parent_id resolves
    for sym in sorted(symbols, key=lambda s: (0 if s.kind == "class" else 1)):
        parent_id: Optional[int] = None
        if sym.scope:
            scope_name = sym.scope.split(":")[-1] if ":" in sym.scope else sym.scope
            # Resolve parent by class name within this file
            for qname, sid in qname_to_id.items():
                if qname.endswith(f".{scope_name}") or qname == scope_name:
                    parent_id = sid
                    break

        sid = graph.insert_symbol(
            file_id, sym.kind, sym.name, sym.qualified_name,
            parent_symbol_id=parent_id,
            signature=sym.signature,
            line_start=sym.line_start,
            line_end=sym.line_end,
            is_async=sym.is_async,
            decorators=sym.decorators,
        )
        qname_to_id[sym.qualified_name] = sid


def _run_ctags(root: Path) -> list:
    """Run ctags on the repo if available; return list of CtagsSymbol."""
    try:
        from .parsers.ctags import parse_repo, is_available
        if not is_available():
            logger.debug("Universal Ctags not available — skipping multi-language parsing")
            return []
        return parse_repo(root)
    except Exception as exc:
        logger.warning("ctags parse failed: %s", exc)
        return []


def _url_to_repo_id(url: str) -> str:
    """Derive a short repo_id slug from a remote URL."""
    import re
    # Extract owner/repo from common URL formats
    m = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1).replace("/", "--")
    return url.split("/")[-1].removesuffix(".git") or "remote"
