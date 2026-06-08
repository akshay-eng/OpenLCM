"""Build the repo orientation block injected into agent context at session start.

Combines:
  - LST structural summary (files, key classes, entry points)
  - Recent LCM session history (what was done before)
  - Available tool hints

Kept compact on purpose — targets ~600 tokens for the full block.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_boot_context(
    lst: Any,
    repo_id: str,
    message_store: Any = None,
    dag: Any = None,
    recent_sessions: int = 3,
) -> str:
    """Return a compact repo orientation string for system message injection.

    Args:
        lst:             LSTGraph instance.
        repo_id:         Which repo to summarize.
        message_store:   MessageStore — used to pull recent session summaries.
        dag:             SummaryDAG — used to pull compressed session history.
        recent_sessions: How many past sessions to include in the history block.
    """
    try:
        stats = lst.get_stats(repo_id, include_repo_meta=True)
    except Exception:
        return ""

    if stats.get("files", 0) == 0:
        return ""

    lines: list[str] = []
    lines.append(f"[OpenLCM Repo Context: {repo_id}]")

    # ── Basic stats ───────────────────────────────────────────────────────
    lang_counts: dict[str, int] = {}
    try:
        with lst._lock:
            rows = lst._conn.execute(
                "SELECT language, COUNT(*) FROM lcm_lst_files WHERE repo_id=? GROUP BY language",
                (repo_id,),
            ).fetchall()
        lang_counts = {r[0]: r[1] for r in rows}
    except Exception:
        pass

    lang_str = " · ".join(f"{l}={n}" for l, n in sorted(lang_counts.items(), key=lambda x: -x[1]))
    lines.append(
        f"Files: {stats['files']:,}  Symbols: {stats['symbols']:,}  Edges: {stats['edges']:,}"
        + (f"  [{lang_str}]" if lang_str else "")
    )

    git = stats.get("git", {})
    if git.get("branch"):
        commit = (git.get("commit_hash") or "")[:12]
        origin = git.get("origin_url", "")
        git_parts = [f"branch:{git['branch']}"]
        if commit:
            git_parts.append(f"commit:{commit}")
        if origin:
            git_parts.append(f"origin:{origin}")
        lines.append("  ".join(git_parts))

    # ── Key classes ───────────────────────────────────────────────────────
    try:
        with lst._lock:
            class_rows = lst._conn.execute(
                """SELECT s.name, f.file_path, s.line_start, s.docstring,
                          (SELECT COUNT(*) FROM lcm_lst_edges e
                           WHERE e.from_symbol_id=s.symbol_id OR e.to_symbol_id=s.symbol_id) AS conn
                   FROM lcm_lst_symbols s
                   JOIN lcm_lst_files f ON s.file_id=f.file_id
                   WHERE f.repo_id=? AND s.kind='class'
                     AND f.file_path NOT LIKE 'docs_%'
                     AND f.file_path NOT LIKE 'tests/%'
                     AND f.file_path NOT LIKE 'test_%'
                     AND f.file_path NOT LIKE '%/test_%'
                     AND f.file_path NOT LIKE '%/tests/%'
                   ORDER BY conn DESC
                   LIMIT 8""",
                (repo_id,),
            ).fetchall()
    except Exception:
        class_rows = []

    if class_rows:
        lines.append("")
        lines.append("Key classes:")
        for name, fpath, line, doc, _ in class_rows:
            doc_str = f" — {doc.split(chr(10))[0][:60]}" if doc else ""
            lines.append(f"  {name:<28} {fpath}:{line}{doc_str}")

    # ── Key entry-point functions (top-level, not methods) ────────────────
    try:
        with lst._lock:
            fn_rows = lst._conn.execute(
                """SELECT s.name, f.file_path, s.line_start
                   FROM lcm_lst_symbols s
                   JOIN lcm_lst_files f ON s.file_id=f.file_id
                   WHERE f.repo_id=? AND s.kind='function' AND s.parent_symbol_id IS NULL
                     AND f.file_path NOT LIKE 'docs_%'
                     AND f.file_path NOT LIKE 'tests/%'
                     AND f.file_path NOT LIKE 'test_%'
                     AND f.file_path NOT LIKE '%/test_%'
                     AND f.file_path NOT LIKE '%/tests/%'
                   ORDER BY (SELECT COUNT(*) FROM lcm_lst_edges e
                              WHERE e.to_symbol_id=s.symbol_id) DESC
                   LIMIT 6""",
                (repo_id,),
            ).fetchall()
    except Exception:
        fn_rows = []

    if fn_rows:
        lines.append("")
        lines.append("Entry-point functions:")
        for name, fpath, line in fn_rows:
            lines.append(f"  {name:<28} {fpath}:{line}")

    # ── Most active files ─────────────────────────────────────────────────
    try:
        with lst._lock:
            file_rows = lst._conn.execute(
                """SELECT f.file_path,
                          COUNT(DISTINCT s.symbol_id) AS sym_count,
                          COUNT(DISTINCT e.edge_id) AS edge_count
                   FROM lcm_lst_files f
                   LEFT JOIN lcm_lst_symbols s ON s.file_id=f.file_id
                   LEFT JOIN lcm_lst_edges e ON e.from_file_id=f.file_id
                   WHERE f.repo_id=?
                   GROUP BY f.file_path
                   ORDER BY sym_count DESC
                   LIMIT 6""",
                (repo_id,),
            ).fetchall()
    except Exception:
        file_rows = []

    if file_rows:
        lines.append("")
        lines.append("Most active files:")
        for fpath, sym_count, edge_count in file_rows:
            lines.append(f"  {fpath:<50} {sym_count} symbols  {edge_count} edges")

    # ── Recent session history ────────────────────────────────────────────
    if dag is not None:
        try:
            history = _get_recent_session_summaries(dag, repo_id, limit=recent_sessions)
            if history:
                lines.append("")
                lines.append("Recent session history:")
                for entry in history:
                    lines.append(f"  • {entry}")
        except Exception:
            pass

    # ── Tool hints ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("Available LST tools (use these instead of reading files):")
    lines.append("  lcm_lst_find(name)            Find any symbol")
    lines.append("  lcm_lst_class(ClassName)       Class + all methods + linked facts")
    lines.append("  lcm_lst_callers(func)          Who calls this function")
    lines.append("  lcm_lst_callees(func)          What this function calls")
    lines.append("  lcm_lst_file(path)             All symbols in a file")
    lines.append("  lcm_read_file(path)            Smart read: compact if already seen, full if first time")
    lines.append("  lcm_remember(key, val, symbol=X)  Pin a discovery to a symbol for next session")
    lines.append("  lcm_lst_facts(symbol)          All facts pinned to a symbol")
    lines.append("[/OpenLCM Repo Context]")

    return "\n".join(lines)


def build_file_context(
    lst: Any,
    file_path: str,
    repo_id: str,
    *,
    facts: Any = None,
) -> str:
    """Return a compact structural summary for a file (used for repeat-read dedup)."""
    try:
        syms = lst.get_file_symbols(file_path, repo_id)
    except Exception:
        return ""

    if not syms:
        return f"[{file_path}: not indexed in LST]"

    lines: list[str] = [f"[LST: {file_path}]"]
    lines.append("Already analyzed this session. Structural summary:")

    for sym in syms:
        kind = sym.get("kind", "")
        if kind in ("import", "variable"):
            continue
        name = sym.get("name", "")
        line = sym.get("line_start", "")
        sig = sym.get("signature", "")
        doc = (sym.get("docstring") or "").split("\n")[0][:60]

        indent = "    " if kind == "method" else "  "
        kind_tag = {"class": "class", "function": "def", "method": "def"}.get(kind, kind)
        sig_part = f" {sig[:80]}" if sig else ""
        doc_part = f" — {doc}" if doc else ""
        line_part = f"  (:{line})" if line else ""
        lines.append(f"{indent}{kind_tag} {name}{sig_part}{line_part}{doc_part}")

    # Surface linked facts if fact store provided
    if facts is not None:
        fname = file_path.replace("\\", "/").split("/")[-1].replace(".py", "")
        tag = f"lst.{repo_id}.{fname}"
        try:
            pinned = facts.recall_query(tag, limit=5)
            if pinned:
                lines.append("")
                lines.append("  Pinned facts from previous sessions:")
                for f in pinned:
                    lines.append(f"    • [{f['category']}] {f['key']}: {f['value'][:120]}")
        except Exception:
            pass

    lines.append(f"[Use lcm_read_file('{file_path}', force_full=true) to get raw content]")
    return "\n".join(lines)


def _get_recent_session_summaries(dag: Any, repo_id: str, limit: int = 3) -> list[str]:
    """Pull top-level summary nodes from recent sessions."""
    summaries: list[str] = []
    try:
        # Get recent sessions from DAG (depth=0 nodes are the root summaries)
        rows = dag._conn.execute(
            """SELECT session_id, summary, latest_at
               FROM summary_nodes
               WHERE depth = 0
               ORDER BY latest_at DESC
               LIMIT ?""",
            (limit * 2,),
        ).fetchall()

        seen_sessions: set[str] = set()
        for session_id, summary, _ in rows:
            if session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            short_id = session_id[:16] if len(session_id) > 16 else session_id
            first_line = (summary or "").split("\n")[0][:120].strip()
            if first_line:
                summaries.append(f"[{short_id}] {first_line}")
            if len(summaries) >= limit:
                break
    except Exception:
        pass
    return summaries
