"""LSTGraph — SQLite-backed Lossless Semantic Tree for codebase graphs.

Stores every file, symbol, and relationship extracted from a repository.
Follows the same pattern as FactStore: thread-safe, single-connection, same db file.
"""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

LST_FORMAT_VERSION = 1


class LSTGraph:
    """Persistent graph of codebase symbols and relationships."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        # networkx cache: repo_id → (symbol_count_when_built, nx.DiGraph)
        self._nx_cache: dict[str, tuple[int, Any]] = {}
        self._init_db()

    # ── Init ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        from ..core.db_bootstrap import configure_connection, run_versioned_migrations
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        configure_connection(self._conn)
        run_versioned_migrations(self._conn)

    # ── Repos ─────────────────────────────────────────────────────────────

    def upsert_repo(
        self,
        repo_id: str,
        origin_url: str = "",
        branch: str = "",
        commit_hash: str = "",
        local_path: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO lcm_lst_repos(repo_id, origin_url, branch, commit_hash, local_path, last_scanned)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    origin_url=excluded.origin_url,
                    branch=excluded.branch,
                    commit_hash=excluded.commit_hash,
                    local_path=excluded.local_path,
                    last_scanned=excluded.last_scanned
                """,
                (repo_id, origin_url, branch, commit_hash, local_path, now),
            )

    def get_repo(self, repo_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM lcm_lst_repos WHERE repo_id=?", (repo_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_repos(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM lcm_lst_repos ORDER BY last_scanned DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Write: files ──────────────────────────────────────────────────────

    def upsert_file(
        self,
        repo_id: str,
        file_path: str,
        language: str,
        file_hash: str,
        line_count: int,
    ) -> int:
        """Insert or replace a file row; return file_id."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO lcm_lst_files(repo_id, file_path, language, file_hash, line_count, last_scanned)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id, file_path) DO UPDATE SET
                    language=excluded.language,
                    file_hash=excluded.file_hash,
                    line_count=excluded.line_count,
                    last_scanned=excluded.last_scanned
                """,
                (repo_id, file_path, language, file_hash, line_count, now),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_file_hash(self, repo_id: str, file_path: str) -> Optional[str]:
        """Return stored hash for a file, or None if not yet scanned."""
        with self._lock:
            row = self._conn.execute(
                "SELECT file_hash FROM lcm_lst_files WHERE repo_id=? AND file_path=?",
                (repo_id, file_path),
            ).fetchone()
            return row["file_hash"] if row else None

    def get_file_id(self, repo_id: str, file_path: str) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT file_id FROM lcm_lst_files WHERE repo_id=? AND file_path=?",
                (repo_id, file_path),
            ).fetchone()
            return row["file_id"] if row else None

    def delete_file_symbols(self, file_id: int) -> None:
        """Delete all symbols and edges for a file (ON DELETE CASCADE handles edges)."""
        with self._lock:
            self._conn.execute("DELETE FROM lcm_lst_symbols WHERE file_id=?", (file_id,))
            self._conn.execute("DELETE FROM lcm_lst_edges WHERE from_file_id=?", (file_id,))

    # ── Write: symbols ────────────────────────────────────────────────────

    def insert_symbol(
        self,
        file_id: int,
        kind: str,
        name: str,
        qualified_name: str,
        *,
        parent_symbol_id: Optional[int] = None,
        signature: str = "",
        docstring: str = "",
        line_start: int = 0,
        line_end: int = 0,
        is_async: bool = False,
        decorators: list[str] | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO lcm_lst_symbols(
                    file_id, kind, name, qualified_name, parent_symbol_id,
                    signature, docstring, line_start, line_end, is_async, decorators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id, kind, name, qualified_name, parent_symbol_id,
                    signature, docstring[:2000], line_start, line_end,
                    1 if is_async else 0,
                    json.dumps(decorators or []),
                ),
            )
            sid = cur.lastrowid
            # Keep FTS index in sync
            self._conn.execute(
                "INSERT INTO lcm_lst_symbols_fts(rowid, name, qualified_name, docstring, signature) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, name, qualified_name, docstring[:2000], signature),
            )
            return sid  # type: ignore[return-value]

    def get_symbol_id_by_qname(self, qualified_name: str, file_id: int) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT symbol_id FROM lcm_lst_symbols WHERE qualified_name=? AND file_id=?",
                (qualified_name, file_id),
            ).fetchone()
            return row["symbol_id"] if row else None

    # ── Write: edges ──────────────────────────────────────────────────────

    def insert_edge(
        self,
        from_file_id: int,
        edge_type: str,
        to_name: str,
        *,
        from_symbol_id: Optional[int] = None,
        to_symbol_id: Optional[int] = None,
        to_file_id: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO lcm_lst_edges(
                    from_symbol_id, from_file_id, to_symbol_id, to_file_id, edge_type, to_name
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (from_symbol_id, from_file_id, to_symbol_id, to_file_id, edge_type, to_name),
            )

    # ── Queries ───────────────────────────────────────────────────────────

    def find_symbol(
        self,
        name: str,
        kind: Optional[str] = None,
        file: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find symbols by name using FTS5 + exact fallback."""
        limit = min(limit, 200)
        with self._lock:
            # Try FTS5 first
            fts_query = name.replace('"', '""')
            try:
                rows = self._conn.execute(
                    """
                    SELECT s.*, f.file_path, f.repo_id
                    FROM lcm_lst_symbols_fts fts
                    JOIN lcm_lst_symbols s ON s.symbol_id = fts.rowid
                    JOIN lcm_lst_files f ON f.file_id = s.file_id
                    WHERE lcm_lst_symbols_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit * 3),
                ).fetchall()
            except Exception:
                rows = []

            # Fallback to LIKE
            if not rows:
                rows = self._conn.execute(
                    """
                    SELECT s.*, f.file_path, f.repo_id
                    FROM lcm_lst_symbols s
                    JOIN lcm_lst_files f ON f.file_id = s.file_id
                    WHERE s.name LIKE ? ESCAPE '\\'
                    LIMIT ?
                    """,
                    (f"%{name}%", limit * 3),
                ).fetchall()

            results = [_row_to_dict(r) for r in rows]

            if kind:
                results = [r for r in results if r.get("kind") == kind]
            if file:
                results = [r for r in results if file in (r.get("file_path") or "")]

            return results[:limit]

    def get_file_symbols(self, file_path: str, repo_id: str = "default") -> list[dict[str, Any]]:
        """All symbols in a file, grouped by kind."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.*
                FROM lcm_lst_symbols s
                JOIN lcm_lst_files f ON f.file_id = s.file_id
                WHERE f.file_path = ? AND f.repo_id = ?
                ORDER BY s.line_start
                """,
                (file_path, repo_id),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    def get_class(self, class_name: str, repo_id: str = "default") -> dict[str, Any]:
        """Return a class symbol with all its methods and base classes."""
        with self._lock:
            # Find class symbol
            cls_rows = self._conn.execute(
                """
                SELECT s.*, f.file_path
                FROM lcm_lst_symbols s
                JOIN lcm_lst_files f ON f.file_id = s.file_id
                WHERE s.name = ? AND s.kind = 'class' AND f.repo_id = ?
                LIMIT 5
                """,
                (class_name, repo_id),
            ).fetchall()
            if not cls_rows:
                return {"error": f"class '{class_name}' not found"}

            cls = _row_to_dict(cls_rows[0])

            # Methods
            methods = self._conn.execute(
                """
                SELECT * FROM lcm_lst_symbols
                WHERE parent_symbol_id = ? AND kind = 'method'
                ORDER BY line_start
                """,
                (cls["symbol_id"],),
            ).fetchall()
            cls["methods"] = [_row_to_dict(m) for m in methods]

            # Base classes (inherits edges)
            bases = self._conn.execute(
                """
                SELECT to_name FROM lcm_lst_edges
                WHERE from_symbol_id = ? AND edge_type = 'inherits'
                """,
                (cls["symbol_id"],),
            ).fetchall()
            cls["bases"] = [r["to_name"] for r in bases]

            return cls

    def get_callers(self, function_name: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find all symbols that call a given function name."""
        limit = min(limit, 200)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT s.*, f.file_path
                FROM lcm_lst_edges e
                JOIN lcm_lst_symbols s ON s.symbol_id = e.from_symbol_id
                JOIN lcm_lst_files f ON f.file_id = s.file_id
                WHERE e.edge_type = 'calls'
                  AND (e.to_name = ? OR e.to_name LIKE ?)
                LIMIT ?
                """,
                (function_name, f"%.{function_name}", limit),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    def get_callees(self, function_name: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find all functions called by a given function."""
        limit = min(limit, 200)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT e.to_name, e.to_symbol_id,
                       s.kind, s.qualified_name, s.signature, s.docstring,
                       f.file_path
                FROM lcm_lst_edges e
                JOIN lcm_lst_symbols caller ON caller.symbol_id = e.from_symbol_id
                LEFT JOIN lcm_lst_symbols s ON s.symbol_id = e.to_symbol_id
                LEFT JOIN lcm_lst_files f ON f.file_id = s.file_id
                WHERE e.edge_type = 'calls'
                  AND (caller.name = ? OR caller.qualified_name LIKE ?)
                LIMIT ?
                """,
                (function_name, f"%.{function_name}", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_imports(self, file_path: str, repo_id: str = "default") -> list[str]:
        """Return list of module names imported by a file."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT e.to_name
                FROM lcm_lst_edges e
                JOIN lcm_lst_files f ON f.file_id = e.from_file_id
                WHERE f.file_path = ? AND f.repo_id = ? AND e.edge_type = 'imports'
                ORDER BY e.to_name
                """,
                (file_path, repo_id),
            ).fetchall()
            return [r["to_name"] for r in rows]

    def get_refs(self, symbol_name: str, limit: int = 30) -> list[dict[str, Any]]:
        """All edges referencing a given name (calls + imports + inherits)."""
        limit = min(limit, 200)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.edge_type, e.to_name,
                       s.name AS from_name, s.qualified_name AS from_qname,
                       s.kind AS from_kind, f.file_path
                FROM lcm_lst_edges e
                JOIN lcm_lst_files f ON f.file_id = e.from_file_id
                LEFT JOIN lcm_lst_symbols s ON s.symbol_id = e.from_symbol_id
                WHERE e.to_name = ? OR e.to_name LIKE ?
                LIMIT ?
                """,
                (symbol_name, f"%.{symbol_name}", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self, repo_id: str = "default", include_repo_meta: bool = False) -> dict[str, Any]:
        with self._lock:
            file_count = self._conn.execute(
                "SELECT COUNT(*) FROM lcm_lst_files WHERE repo_id=?", (repo_id,)
            ).fetchone()[0]
            sym_count = self._conn.execute(
                """
                SELECT COUNT(*) FROM lcm_lst_symbols s
                JOIN lcm_lst_files f ON f.file_id = s.file_id
                WHERE f.repo_id = ?
                """,
                (repo_id,),
            ).fetchone()[0]
            edge_count = self._conn.execute(
                """
                SELECT COUNT(*) FROM lcm_lst_edges e
                JOIN lcm_lst_files f ON f.file_id = e.from_file_id
                WHERE f.repo_id = ?
                """,
                (repo_id,),
            ).fetchone()[0]
            result: dict[str, Any] = {
                "repo_id": repo_id,
                "files": file_count,
                "symbols": sym_count,
                "edges": edge_count,
            }
            if include_repo_meta:
                repo = self._conn.execute(
                    "SELECT * FROM lcm_lst_repos WHERE repo_id=?", (repo_id,)
                ).fetchone()
                if repo:
                    result["git"] = dict(repo)
            return result

    def list_files(self, repo_id: str = "default", limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT file_id, file_path, language, line_count, last_scanned
                FROM lcm_lst_files
                WHERE repo_id = ?
                ORDER BY file_path
                LIMIT ?
                """,
                (repo_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── NetworkX in-memory graph ──────────────────────────────────────────

    def build_nx_graph(self, repo_id: str = "default") -> Any:
        """
        Load all symbols and resolved edges into a networkx DiGraph.

        Nodes: symbol_id (int), attrs: name, kind, qualified_name, file_path
        Edges: (from_symbol_id, to_symbol_id), attrs: edge_type

        Only resolved edges (where to_symbol_id is NOT NULL) are included.
        Requires: pip install networkx
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("pip install networkx to use graph traversal tools") from exc

        G: Any = nx.DiGraph()
        with self._lock:
            # Load all symbols for this repo as nodes
            rows = self._conn.execute(
                """
                SELECT s.symbol_id, s.name, s.kind, s.qualified_name, f.file_path
                FROM lcm_lst_symbols s
                JOIN lcm_lst_files f ON f.file_id = s.file_id
                WHERE f.repo_id = ?
                """,
                (repo_id,),
            ).fetchall()
            for r in rows:
                G.add_node(
                    r["symbol_id"],
                    name=r["name"],
                    kind=r["kind"],
                    qualified_name=r["qualified_name"],
                    file_path=r["file_path"],
                )

            # Load resolved edges (to_symbol_id not null)
            edges = self._conn.execute(
                """
                SELECT e.from_symbol_id, e.to_symbol_id, e.edge_type
                FROM lcm_lst_edges e
                JOIN lcm_lst_files f ON f.file_id = e.from_file_id
                WHERE f.repo_id = ?
                  AND e.from_symbol_id IS NOT NULL
                  AND e.to_symbol_id IS NOT NULL
                """,
                (repo_id,),
            ).fetchall()
            for e in edges:
                G.add_edge(e["from_symbol_id"], e["to_symbol_id"], edge_type=e["edge_type"])

        return G

    def _get_nx(self, repo_id: str = "default") -> Any:
        """Return cached nx.DiGraph, rebuilding only when symbol count changes."""
        try:
            import networkx as nx  # noqa: F401
        except ImportError:
            return None

        stats = self.get_stats(repo_id)
        current_count = stats["symbols"]
        cached = self._nx_cache.get(repo_id)
        if cached and cached[0] == current_count:
            return cached[1]

        G = self.build_nx_graph(repo_id)
        self._nx_cache[repo_id] = (current_count, G)
        return G

    def _symbol_id_for(self, name: str, repo_id: str = "default") -> Optional[int]:
        """Find the most likely symbol_id for a short name (prefers functions/methods)."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.symbol_id, s.kind FROM lcm_lst_symbols s
                JOIN lcm_lst_files f ON f.file_id = s.file_id
                WHERE s.name = ? AND f.repo_id = ?
                ORDER BY CASE s.kind
                    WHEN 'function' THEN 0
                    WHEN 'method'   THEN 1
                    WHEN 'class'    THEN 2
                    ELSE 3
                END
                LIMIT 1
                """,
                (name, repo_id),
            ).fetchone()
            return rows["symbol_id"] if rows else None

    def get_path(
        self,
        from_name: str,
        to_name: str,
        repo_id: str = "default",
    ) -> dict[str, Any]:
        """Find the shortest call path between two symbols using networkx."""
        G = self._get_nx(repo_id)
        if G is None:
            return {"error": "networkx not installed — pip install networkx"}

        src = self._symbol_id_for(from_name, repo_id)
        dst = self._symbol_id_for(to_name, repo_id)
        if src is None:
            return {"error": f"symbol '{from_name}' not found"}
        if dst is None:
            return {"error": f"symbol '{to_name}' not found"}

        try:
            import networkx as nx
            path_ids = nx.shortest_path(G, src, dst)
        except Exception:
            return {"path": [], "found": False, "message": f"No path from '{from_name}' to '{to_name}'"}

        path_nodes = []
        for nid in path_ids:
            attrs = G.nodes.get(nid, {})
            path_nodes.append({
                "symbol_id": nid,
                "name": attrs.get("name", "?"),
                "kind": attrs.get("kind", "?"),
                "qualified_name": attrs.get("qualified_name", ""),
                "file_path": attrs.get("file_path", ""),
            })
        return {"path": path_nodes, "found": True, "hops": len(path_nodes) - 1}

    def get_ancestors(
        self,
        symbol_name: str,
        depth: int = 5,
        repo_id: str = "default",
    ) -> dict[str, Any]:
        """All symbols that transitively call symbol_name (up to depth hops)."""
        G = self._get_nx(repo_id)
        if G is None:
            return {"error": "networkx not installed — pip install networkx"}

        sid = self._symbol_id_for(symbol_name, repo_id)
        if sid is None:
            return {"error": f"symbol '{symbol_name}' not found"}

        try:
            import networkx as nx
            # Reverse graph to find who calls sid
            RG = G.reverse(copy=False)
            reachable = set()
            frontier = {sid}
            for _ in range(depth):
                next_frontier = set()
                for node in frontier:
                    for pred in RG.successors(node):
                        if pred not in reachable and pred != sid:
                            next_frontier.add(pred)
                reachable.update(next_frontier)
                frontier = next_frontier
                if not frontier:
                    break
        except Exception as exc:
            return {"error": str(exc)}

        nodes = []
        for nid in reachable:
            attrs = G.nodes.get(nid, {})
            nodes.append({
                "symbol_id": nid,
                "name": attrs.get("name", "?"),
                "kind": attrs.get("kind", "?"),
                "qualified_name": attrs.get("qualified_name", ""),
                "file_path": attrs.get("file_path", ""),
            })
        nodes.sort(key=lambda x: x["name"])
        return {"symbol": symbol_name, "total": len(nodes), "ancestors": nodes}

    def get_descendants(
        self,
        symbol_name: str,
        depth: int = 5,
        repo_id: str = "default",
    ) -> dict[str, Any]:
        """All symbols transitively called by symbol_name (up to depth hops)."""
        G = self._get_nx(repo_id)
        if G is None:
            return {"error": "networkx not installed — pip install networkx"}

        sid = self._symbol_id_for(symbol_name, repo_id)
        if sid is None:
            return {"error": f"symbol '{symbol_name}' not found"}

        try:
            reachable = set()
            frontier = {sid}
            for _ in range(depth):
                next_frontier = set()
                for node in frontier:
                    for succ in G.successors(node):
                        if succ not in reachable and succ != sid:
                            next_frontier.add(succ)
                reachable.update(next_frontier)
                frontier = next_frontier
                if not frontier:
                    break
        except Exception as exc:
            return {"error": str(exc)}

        nodes = []
        for nid in reachable:
            attrs = G.nodes.get(nid, {})
            nodes.append({
                "symbol_id": nid,
                "name": attrs.get("name", "?"),
                "kind": attrs.get("kind", "?"),
                "qualified_name": attrs.get("qualified_name", ""),
                "file_path": attrs.get("file_path", ""),
            })
        nodes.sort(key=lambda x: x["name"])
        return {"symbol": symbol_name, "total": len(nodes), "descendants": nodes}

    # ── Portable context export/import ────────────────────────────────────

    def export_graph(
        self,
        output_path: str | Path,
        repo_id: str = "default",
        repo_path: str = "",
        compress: bool = True,
    ) -> dict[str, Any]:
        """
        Export the LST graph for a repo as a portable .lcmgraph file.

        The exported file is self-contained JSON (optionally gzipped) and can be
        imported into any LSTGraph instance on any machine or by any agent.

        Args:
            output_path: destination file path (e.g. "myapp.lcmgraph")
            repo_id: which repo to export (default "default")
            repo_path: informational only — recorded in the export header
            compress: if True, gzip-compress the output (default True)
        """
        out = Path(output_path)
        with self._lock:
            files = self._conn.execute(
                "SELECT file_path, language, line_count, file_hash FROM lcm_lst_files WHERE repo_id=?",
                (repo_id,),
            ).fetchall()

            symbols = self._conn.execute(
                """
                SELECT s.kind, s.name, s.qualified_name, s.signature, s.docstring,
                       s.line_start, s.line_end, s.is_async, s.decorators,
                       f.file_path AS file_path,
                       ps.qualified_name AS parent_qname
                FROM lcm_lst_symbols s
                JOIN lcm_lst_files f ON f.file_id = s.file_id
                LEFT JOIN lcm_lst_symbols ps ON ps.symbol_id = s.parent_symbol_id
                WHERE f.repo_id = ?
                ORDER BY s.symbol_id
                """,
                (repo_id,),
            ).fetchall()

            edges = self._conn.execute(
                """
                SELECT e.edge_type, e.to_name,
                       fs.qualified_name AS from_qname,
                       ff.file_path AS from_file
                FROM lcm_lst_edges e
                JOIN lcm_lst_files ff ON ff.file_id = e.from_file_id
                LEFT JOIN lcm_lst_symbols fs ON fs.symbol_id = e.from_symbol_id
                WHERE ff.repo_id = ?
                ORDER BY e.edge_id
                """,
                (repo_id,),
            ).fetchall()

        payload = {
            "version": LST_FORMAT_VERSION,
            "type": "lcmgraph",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "repo_id": repo_id,
            "repo_path": repo_path,
            "stats": {
                "files": len(files),
                "symbols": len(symbols),
                "edges": len(edges),
            },
            "files": [dict(r) for r in files],
            "symbols": [
                {
                    "kind": r["kind"],
                    "name": r["name"],
                    "qualified_name": r["qualified_name"],
                    "file_path": r["file_path"],
                    "parent_qname": r["parent_qname"] or "",
                    "signature": r["signature"],
                    "docstring": r["docstring"],
                    "line_start": r["line_start"],
                    "line_end": r["line_end"],
                    "is_async": bool(r["is_async"]),
                    "decorators": json.loads(r["decorators"] or "[]"),
                }
                for r in symbols
            ],
            "edges": [
                {
                    "from_qname": r["from_qname"] or r["from_file"] or "",
                    "edge_type": r["edge_type"],
                    "to_name": r["to_name"],
                }
                for r in edges
            ],
        }

        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with gzip.open(str(out), "wb") as fh:
                fh.write(raw)
        else:
            out.write_bytes(raw)

        size_kb = out.stat().st_size / 1024
        return {
            "path": str(out),
            "compressed": compress,
            "size_kb": round(size_kb, 1),
            **payload["stats"],
        }

    def import_graph(
        self,
        input_path: str | Path,
        target_repo_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Import an .lcmgraph file into this LSTGraph database.

        Args:
            input_path: path to the .lcmgraph file (gzipped or plain JSON)
            target_repo_id: override the repo_id from the file (default: use file's repo_id)
        """
        inp = Path(input_path)
        if not inp.exists():
            return {"error": f"file not found: {inp}"}

        raw = inp.read_bytes()
        if raw[:2] == b"\x1f\x8b":  # gzip magic bytes
            raw = gzip.decompress(raw)

        try:
            payload = json.loads(raw)
        except Exception as exc:
            return {"error": f"invalid format: {exc}"}

        if payload.get("type") != "lcmgraph":
            return {"error": "not an lcmgraph file — missing type field"}

        repo_id = target_repo_id or payload.get("repo_id", "default")
        files_data = payload.get("files", [])
        symbols_data = payload.get("symbols", [])
        edges_data = payload.get("edges", [])

        # Index: file_path → file_id
        file_ids: dict[str, int] = {}
        for f in files_data:
            fid = self.upsert_file(
                repo_id,
                f["file_path"],
                f.get("language", "python"),
                f.get("file_hash", ""),
                f.get("line_count", 0),
            )
            self.delete_file_symbols(fid)
            file_ids[f["file_path"]] = fid

        # Index: qualified_name → symbol_id (built as we insert)
        qname_to_id: dict[str, int] = {}

        # First pass: insert all symbols
        for sym in symbols_data:
            fp = sym.get("file_path", "")
            fid = file_ids.get(fp)
            if fid is None:
                continue
            parent_id = qname_to_id.get(sym.get("parent_qname", ""))
            sid = self.insert_symbol(
                fid,
                sym["kind"],
                sym["name"],
                sym["qualified_name"],
                parent_symbol_id=parent_id,
                signature=sym.get("signature", ""),
                docstring=sym.get("docstring", ""),
                line_start=sym.get("line_start", 0),
                line_end=sym.get("line_end", 0),
                is_async=sym.get("is_async", False),
                decorators=sym.get("decorators", []),
            )
            qname_to_id[sym["qualified_name"]] = sid

        # Build symbol_id → file_id mapping for fast edge resolution
        sid_to_fid: dict[int, int] = {}
        for fp, fid in file_ids.items():
            with self._lock:
                rows = self._conn.execute(
                    "SELECT symbol_id FROM lcm_lst_symbols WHERE file_id=?", (fid,)
                ).fetchall()
            for r in rows:
                sid_to_fid[r[0]] = fid

        # Second pass: insert edges
        inserted_edges = 0
        for edge in edges_data:
            from_qname = edge.get("from_qname", "")
            from_symbol_id = qname_to_id.get(from_qname)

            # Resolve from_file_id
            from_file_id = None
            if from_symbol_id is not None:
                from_file_id = sid_to_fid.get(from_symbol_id)
            if from_file_id is None:
                # from_qname may be a file path (for file-level import edges)
                from_file_id = file_ids.get(from_qname)
            if from_file_id is None:
                continue

            self.insert_edge(
                from_file_id,
                edge["edge_type"],
                edge["to_name"],
                from_symbol_id=from_symbol_id,
            )
            inserted_edges += 1

        # Invalidate nx cache for this repo
        self._nx_cache.pop(repo_id, None)

        return {
            "repo_id": repo_id,
            "files": len(file_ids),
            "symbols": len(qname_to_id),
            "edges": inserted_edges,
            "source_version": payload.get("version"),
            "exported_at": payload.get("exported_at", ""),
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if "decorators" in d and isinstance(d["decorators"], str):
        try:
            d["decorators"] = json.loads(d["decorators"])
        except Exception:
            d["decorators"] = []
    return d
