"""Universal Ctags parser — multi-language symbol extraction via subprocess.

Universal Ctags (https://ctags.io) supports 100+ languages including
TypeScript, JavaScript, Go, Java, Rust, Ruby, C/C++, PHP, Swift, Kotlin, etc.

Install:
    macOS:   brew install universal-ctags
    Ubuntu:  sudo apt install universal-ctags
    Windows: scoop install universal-ctags

The `--output-format=json` flag outputs one JSON object per tag.
We run ctags ONCE per repo scan (not per file) for maximum speed.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Languages handled by the Python AST parser — skip ctags for these
# since AST gives richer data (docstrings, full signatures, call edges).
_SKIP_LANGUAGES = {"Python"}

# ctags kind → LST kind mapping
_KIND_MAP = {
    # Functions / methods
    "function":    "function",
    "f":           "function",
    "method":      "method",
    "m":           "method",
    "procedure":   "function",
    "subroutine":  "function",
    # Classes / structs / interfaces
    "class":       "class",
    "c":           "class",
    "struct":      "class",
    "s":           "class",
    "interface":   "class",
    "i":           "class",
    "type":        "class",
    "t":           "class",
    "trait":       "class",
    # Variables / constants / properties
    "variable":    "variable",
    "v":           "variable",
    "constant":    "variable",
    "field":       "variable",
    "property":    "variable",
    "member":      "variable",
    # Imports / modules
    "import":      "import",
    "namespace":   "variable",
    "module":      "variable",
}

# File extensions that ctags can parse (beyond Python)
_CTAGS_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go",
    ".java",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt", ".kts",
    ".c", ".h",
    ".cpp", ".cc", ".cxx", ".hpp",
    ".cs",
    ".scala",
    ".sh", ".bash",
}


@dataclass
class CtagsSymbol:
    kind: str
    name: str
    qualified_name: str
    file_path: str
    language: str
    signature: str = ""
    scope: str = ""          # enclosing scope (class name for methods)
    line_start: int = 0
    line_end: int = 0
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)


def is_available() -> bool:
    """Return True if Universal Ctags is installed and supports JSON output."""
    ctags = _ctags_bin()
    if not ctags:
        return False
    try:
        result = subprocess.run(
            [ctags, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        # Universal Ctags reports "Universal Ctags" in version string
        # Exuberant Ctags does NOT support --output-format=json
        return "Universal Ctags" in result.stdout or "Universal Ctags" in result.stderr
    except Exception:
        return False


def parse_repo(
    repo_path: str | Path,
    languages: Optional[list[str]] = None,
) -> list[CtagsSymbol]:
    """
    Run Universal Ctags on an entire repo and return all symbols.

    Skips Python files (handled by PythonASTParser with richer output).
    Falls back to empty list if ctags is not installed.

    Args:
        repo_path: root of the repository
        languages: explicit list of language names to parse (None = all supported)
    """
    ctags = _ctags_bin()
    if not ctags:
        logger.debug("ctags not found — skipping multi-language parsing")
        return []

    root = Path(repo_path).resolve()
    cmd = [
        ctags,
        "--recurse",
        "--output-format=json",
        "--fields=+{line}{end}{typeref}{scope}{signature}{language}",
        "--extras=+q",       # include qualified tags
        "--sort=no",         # faster
        "--links=no",
        # Exclude Python (AST parser handles it better)
        "--languages=-Python",
    ]

    if languages:
        cmd += [f"--languages={','.join(languages)}"]

    cmd.append(str(root))

    logger.info("Running ctags on %s …", root)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(root),
        )
    except subprocess.TimeoutExpired:
        logger.warning("ctags timed out on %s", root)
        return []
    except Exception as exc:
        logger.warning("ctags failed: %s", exc)
        return []

    if result.returncode not in (0, 1):  # ctags returns 1 on parse errors but still outputs tags
        logger.warning("ctags exit %d: %s", result.returncode, result.stderr[:200])

    symbols: list[CtagsSymbol] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            tag = json.loads(line)
        except json.JSONDecodeError:
            continue

        tag_type = tag.get("_type", "")
        if tag_type != "tag":
            continue

        raw_kind = tag.get("kind", "").lower()
        kind = _KIND_MAP.get(raw_kind)
        if kind is None:
            continue  # skip unknown kinds (e.g. labels, macros)

        name = tag.get("name", "").strip()
        if not name:
            continue

        path = tag.get("path", "")
        language = tag.get("language", "unknown")

        # Make path repo-relative
        try:
            rel = Path(path).relative_to(root).as_posix()
        except ValueError:
            rel = path

        # Build qualified name from scope
        scope = tag.get("scope", "")
        if scope:
            # ctags scope format: "ClassName" or "ClassName.method"
            scope_name = scope.split(":")[-1] if ":" in scope else scope
            qualified_name = f"{scope_name}.{name}"
        else:
            # Use module-style qualified name
            module = _path_to_module(rel)
            qualified_name = f"{module}.{name}" if module else name

        line_start = int(tag.get("line", 0))
        line_end = int(tag.get("end", line_start))
        signature = tag.get("signature", "")

        symbols.append(CtagsSymbol(
            kind=kind,
            name=name,
            qualified_name=qualified_name,
            file_path=rel,
            language=language,
            signature=signature,
            scope=scope,
            line_start=line_start,
            line_end=line_end,
        ))

    logger.info("ctags extracted %d symbols from %s", len(symbols), root)
    return symbols


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ctags_bin() -> Optional[str]:
    """Find the Universal Ctags binary (cross-platform)."""
    import sys
    names = ["ctags", "universal-ctags", "uctags"]
    # On Windows, scoop installs as ctags.exe; also check common install paths
    if sys.platform == "win32":
        names = ["ctags.exe", "ctags"] + names
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def _path_to_module(file_path: str) -> str:
    """Convert a file path to a dotted module-like name."""
    p = Path(file_path)
    if p.name in ("index.ts", "index.js", "index.tsx", "index.jsx", "mod.rs", "__init__.py"):
        parts = list(p.parent.parts)
    else:
        parts = list(p.parent.parts) + [p.stem]
    # Drop leading path segments that look like build/src roots
    skip = {"src", "lib", "pkg", "app", "internal", ""}
    trimmed = []
    for part in reversed(parts):
        if part in skip:
            break
        trimmed.insert(0, part)
    return ".".join(trimmed) if trimmed else p.stem
