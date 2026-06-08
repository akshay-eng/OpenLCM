"""OpenLCM CLI — openlcm command-line tool.

All read-only commands work without a running engine (just a DB path).
The viz command starts the live dashboard server.

Usage:
  openlcm status [--db PATH]
  openlcm grep QUERY [--limit N] [--db PATH]
  openlcm sessions [--db PATH]
  openlcm expand [--node-id N | --store-id N] [--db PATH]
  openlcm export SESSION_ID [--out FILE] [--db PATH]
  openlcm doctor [--db PATH]
  openlcm viz [--host HOST] [--port PORT] [--no-browser] [--db PATH]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print("Install typer: pip install openlcm[viz]  or  pip install typer")
    sys.exit(1)

app = typer.Typer(
    name="openlcm",
    help="OpenLCM — Lossless Context Management CLI",
    add_completion=False,
)

scan_app = typer.Typer(
    name="scan",
    help="Manage the Lossless Semantic Tree (LST) codebase graph.",
    add_completion=False,
)
app.add_typer(scan_app, name="scan")

_DEFAULT_DB = str(Path.home() / ".openlcm" / "lcm.db")


def _db_option():
    return typer.Option("", "--db", help="Path to lcm.db (default: ~/.openlcm/lcm.db)")


def _resolve_db(db: str) -> Path:
    return Path(db).expanduser().resolve() if db else Path.home() / ".openlcm" / "lcm.db"


def _open_store(db: str):
    from openlcm.core.store import MessageStore
    from openlcm.core.config import LCMConfig
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo(f"Database not found: {db_path}", err=True)
        raise typer.Exit(1)
    config = LCMConfig()
    return MessageStore(db_path, ingest_protection_config=config)


def _open_dag(db: str):
    from openlcm.core.dag import SummaryDAG
    return SummaryDAG(_resolve_db(db))


def _open_lifecycle(db: str):
    from openlcm.core.lifecycle_state import LifecycleStateStore
    return LifecycleStateStore(_resolve_db(db))


# ── status ────────────────────────────────────────────────────────────────

@app.command()
def status(db: str = _db_option()):
    """Show current LCM database status."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo(f"No database at {db_path}. Run an agent with LCM first.")
        raise typer.Exit(0)

    store = _open_store(db)
    dag = _open_dag(db)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()
    total_sessions = row[0] if row else 0
    row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
    total_messages = row[0] if row else 0
    row = conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()
    total_nodes = row[0] if row else 0
    conn.close()

    db_size = db_path.stat().st_size if db_path.exists() else 0

    typer.echo(f"\nOpenLCM Status")
    typer.echo(f"  Database:       {db_path}")
    typer.echo(f"  Size:           {db_size / 1024:.1f} KB")
    typer.echo(f"  Sessions:       {total_sessions}")
    typer.echo(f"  Messages:       {total_messages}")
    typer.echo(f"  DAG nodes:      {total_nodes}")
    typer.echo("")


# ── sessions ──────────────────────────────────────────────────────────────

@app.command()
def sessions(db: str = _db_option()):
    """List all sessions in the database."""
    import sqlite3
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo(f"No database at {db_path}.")
        raise typer.Exit(0)

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT session_id, COUNT(*) as cnt, MIN(timestamp), MAX(timestamp) "
        "FROM messages GROUP BY session_id ORDER BY MAX(timestamp) DESC LIMIT 50"
    ).fetchall()
    conn.close()

    if not rows:
        typer.echo("No sessions found.")
        return

    typer.echo(f"\n{'SESSION ID':<50} {'MSGS':>6}  {'LAST ACTIVE':<20}")
    typer.echo("─" * 80)
    for r in rows:
        import datetime
        last = datetime.datetime.fromtimestamp(r[3]).strftime("%Y-%m-%d %H:%M") if r[3] else "—"
        typer.echo(f"{r[0]:<50} {r[1]:>6}  {last:<20}")
    typer.echo("")


# ── grep ──────────────────────────────────────────────────────────────────

@app.command()
def grep(
    query: str = typer.Argument(..., help="Search query (FTS5 syntax)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    session: str = typer.Option("", "--session", help="Restrict to session ID"),
    db: str = _db_option(),
):
    """Search conversation history with FTS5."""
    store = _open_store(db)
    session_id = session or None
    hits = store.search(query, session_id=session_id, limit=limit, sort="recency")
    if not hits:
        typer.echo(f"No results for '{query}'")
        return
    typer.echo(f"\n{len(hits)} result(s) for '{query}':\n")
    for h in hits:
        import datetime
        ts = h.get("timestamp", 0)
        time_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
        typer.echo(f"  [{h.get('role','?').upper()}] #{h.get('store_id','')}  {time_str}")
        snippet = (h.get("snippet") or h.get("content") or "")[:200].replace("\n", " ")
        typer.echo(f"  {snippet}")
        typer.echo("")


# ── expand ────────────────────────────────────────────────────────────────

@app.command()
def expand(
    node_id: Optional[int] = typer.Option(None, "--node-id", help="Expand a summary node"),
    store_id: Optional[int] = typer.Option(None, "--store-id", help="Expand a raw message by store_id"),
    db: str = _db_option(),
):
    """Expand a summary node or raw message."""
    if node_id is None and store_id is None:
        typer.echo("Provide --node-id or --store-id", err=True)
        raise typer.Exit(1)

    if store_id is not None:
        store = _open_store(db)
        msg = store.get(store_id)
        if not msg:
            typer.echo(f"No message with store_id={store_id}")
            raise typer.Exit(1)
        typer.echo(f"\n[{msg.get('role','?').upper()}] store_id={store_id}  session={msg.get('session_id','')}\n")
        typer.echo(msg.get("content", "") or "(no content)")
        typer.echo("")
        return

    dag = _open_dag(db)
    node = dag.get_node(node_id)
    if not node:
        typer.echo(f"No DAG node with node_id={node_id}")
        raise typer.Exit(1)
    typer.echo(f"\nNode #{node_id} D{node.depth}  tokens={node.token_count}  src={node.source_token_count}\n")
    typer.echo(node.summary)
    typer.echo("")


# ── export ────────────────────────────────────────────────────────────────

@app.command()
def export(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    out: str = typer.Option("", "--out", "-o", help="Output file path (default: session_id.json)"),
    db: str = _db_option(),
):
    """Export full conversation history for a session as JSON."""
    store = _open_store(db)
    dag = _open_dag(db)

    messages = store.get_session_messages(session_id, limit=50000)
    nodes = dag.get_session_nodes(session_id, limit=10000)

    payload = {
        "session_id": session_id,
        "message_count": len(messages),
        "dag_node_count": len(nodes),
        "messages": messages,
        "dag_nodes": [
            {
                "node_id": n.node_id, "depth": n.depth, "summary": n.summary,
                "token_count": n.token_count, "source_token_count": n.source_token_count,
                "source_ids": n.source_ids, "source_type": n.source_type,
                "created_at": n.created_at, "earliest_at": n.earliest_at,
                "latest_at": n.latest_at, "expand_hint": n.expand_hint,
            }
            for n in nodes
        ],
    }

    output_path = out or f"{session_id[:30].replace('/', '-')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    typer.echo(f"Exported {len(messages)} messages + {len(nodes)} DAG nodes → {output_path}")


# ── doctor ────────────────────────────────────────────────────────────────

@app.command()
def doctor(db: str = _db_option()):
    """Run database integrity checks."""
    import sqlite3
    db_path = _resolve_db(db)
    typer.echo(f"\nOpenLCM Doctor — {db_path}\n")

    if not db_path.exists():
        typer.echo("✗ Database file not found")
        raise typer.Exit(1)

    conn = sqlite3.connect(str(db_path))
    checks = []

    # Quick check
    row = conn.execute("PRAGMA quick_check").fetchone()
    ok = row and row[0] == "ok"
    checks.append(("SQLite integrity", "ok" if ok else "FAIL", ok))

    # Table existence
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in ("messages", "summary_nodes"):
        checks.append((f"Table: {t}", "present" if t in tables else "MISSING", t in tables))

    # FTS
    fts_ok = "messages_fts" in tables and "nodes_fts" in tables
    checks.append(("FTS5 indexes", "present" if fts_ok else "MISSING", fts_ok))

    # Row counts
    msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] if "messages" in tables else 0
    nodes = conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] if "summary_nodes" in tables else 0
    checks.append(("Message rows", str(msgs), True))
    checks.append(("DAG node rows", str(nodes), True))

    conn.close()

    for name, result, ok in checks:
        icon = "✓" if ok else "✗"
        typer.echo(f"  {icon} {name}: {result}")

    typer.echo(f"\n  DB size: {db_path.stat().st_size / 1024:.1f} KB\n")


# ── viz ───────────────────────────────────────────────────────────────────

@app.command()
def viz(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(7842, "--port", "-p", help="Bind port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
    db: str = _db_option(),
):
    """Start the live OpenLCM visualization dashboard."""
    try:
        from openlcm.viz.server import create_app, serve
    except ImportError:
        typer.echo("Install visualization deps: pip install openlcm[viz]", err=True)
        raise typer.Exit(1)

    db_path = _resolve_db(db)

    # Create a read-only engine for the dashboard (no summarization needed)
    engine = None
    if db_path.exists():
        try:
            from openlcm.core.engine import LCMEngine
            from openlcm.backends.base import SummaryBackend

            class _NullBackend(SummaryBackend):
                async def summarize(self, prompt, max_tokens, model="", timeout=None):
                    return None

            engine = LCMEngine(backend=_NullBackend(), db_path=str(db_path))
        except Exception as exc:
            typer.echo(f"Warning: could not initialize engine: {exc}", err=True)

    app = create_app(engine)
    serve(app, host=host, port=port, open_browser=not no_browser)


# ── scan subcommands ─────────────────────────────────────────────────────────

def _open_lst(db: str) -> "LSTGraph":
    from openlcm.code.graph import LSTGraph
    return LSTGraph(_resolve_db(db))


def _scan_with_progress(scanner, path: str, graph, kwargs: dict) -> dict:
    """Run scanner.scan() with a rich live progress display."""
    try:
        from rich.progress import (
            Progress, SpinnerColumn, TextColumn,
            MofNCompleteColumn, TimeElapsedColumn,
        )
        from rich.console import Console
        _rich = True
    except ImportError:
        _rich = False

    if not _rich:
        return scanner.scan(path, graph, **kwargs)

    console = Console()
    parsed = 0
    skipped = 0
    _status_text = ["Initializing..."]

    def _truncate(s: str, n: int = 55) -> str:
        return s if len(s) <= n else "…" + s[-(n - 1):]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(_status_text[0], total=None)

        def on_progress(event: str, data: dict) -> None:
            nonlocal parsed, skipped
            if event == "clone_start":
                url = data.get("url", "")
                progress.update(task, description=f"Cloning {_truncate(url, 60)}...")
            elif event == "clone_done":
                cached = " (cached)" if data.get("was_cached") else " (fresh)"
                progress.update(task, description=f"Cloned{cached}")
            elif event == "phase_start":
                phase = data.get("phase", "")
                if phase == "python":
                    progress.update(task, description="Scanning Python files...")
                elif phase == "ctags":
                    progress.update(task, description="Running ctags (multi-language)...")
            elif event == "file_scan":
                parsed += 1
                lang = data.get("lang", "")
                rel = _truncate(data.get("path", ""))
                progress.update(
                    task,
                    description=f"[cyan]{lang}[/]  {rel}  "
                                f"[dim]parsed={parsed}  skipped={skipped}[/]",
                )
            elif event == "file_skip":
                skipped += 1
                rel = _truncate(data.get("path", ""))
                progress.update(
                    task,
                    description=f"[dim]skip  {rel}  parsed={parsed}  skipped={skipped}[/]",
                )
            elif event == "file_error":
                rel = _truncate(data.get("path", ""))
                progress.update(task, description=f"[red]error[/]  {rel}")
            elif event == "ctags_files":
                n = data.get("count", 0)
                progress.update(task, description=f"ctags: processing {n} files...")

        stats = scanner.scan(path, graph, **kwargs, on_progress=on_progress)

    return stats


@scan_app.command("repo")
def scan_repo(
    path: str = typer.Argument(..., help="Local path or remote URL (GitHub, GitLab, etc.)"),
    repo_id: str = typer.Option("", "--repo-id", "-r", help="Logical repo identifier (auto-derived from URL if omitted)"),
    db: str = _db_option(),
    force: bool = typer.Option(False, "--force", "-f", help="Re-parse all files (ignore hash cache)"),
    branch: str = typer.Option("", "--branch", "-b", help="Branch or tag to clone (default: remote HEAD)"),
    depth: int = typer.Option(1, "--depth", help="Clone depth for remote repos (0 = full history)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
):
    """
    Scan a repository and build the Lossless Semantic Tree (LST).

    Accepts local paths or remote URLs:

        openlcm scan repo /path/to/myapp
        openlcm scan repo https://github.com/user/repo
        openlcm scan repo https://github.com/user/repo --branch develop
        openlcm scan repo git@github.com:user/repo.git --repo-id myrepo

    Remote repos are cloned to ~/.openlcm/repos/ and cached — repeated scans
    only re-parse files that changed.
    """
    from openlcm.code.graph import LSTGraph
    from openlcm.code.scanner import RepoScanner
    from openlcm.code.git import is_url

    db_path = _resolve_db(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    is_remote = is_url(path)

    if not quiet:
        typer.echo(f"\nOpenLCM LST Scan")
        typer.echo(f"  Source:   {'[remote] ' if is_remote else ''}{path}")
        if repo_id:
            typer.echo(f"  Repo ID:  {repo_id}")
        typer.echo(f"  Database: {db_path}")
        typer.echo(f"  Mode:     {'force (re-parse all)' if force else 'incremental'}")
        if is_remote:
            typer.echo(f"  Cache:    ~/.openlcm/repos/")
        typer.echo("")

    graph = LSTGraph(str(db_path))
    scanner = RepoScanner()

    kwargs: dict = dict(
        repo_id=repo_id or "default",
        force=force,
    )
    if is_remote:
        kwargs["branch"] = branch or None
        kwargs["clone_depth"] = depth

    if not quiet:
        stats = _scan_with_progress(scanner, path, graph, kwargs)
    else:
        stats = scanner.scan(path, graph, **kwargs)

    if "error" in stats:
        typer.echo(f"Error: {stats['error']}", err=True)
        raise typer.Exit(1)

    # Show language breakdown
    langs = stats.get("languages", {})
    lang_str = "  ".join(f"{lang}={n}" for lang, n in sorted(langs.items(), key=lambda x: -x[1]))

    typer.echo(f"  Scanned:   {stats['scanned']} files  ({stats['skipped']} skipped, {stats.get('errors',0)} errors)")
    typer.echo(f"  Languages: {lang_str or '—'}")
    typer.echo(f"  Symbols:   {stats['symbols']:,}")
    typer.echo(f"  Edges:     {stats['edges']:,}")
    typer.echo(f"  Time:      {stats['elapsed_s']:.2f}s")

    git = stats.get("git", {})
    if git.get("commit_hash"):
        typer.echo(f"\n  Git:")
        typer.echo(f"    Branch:  {git.get('branch','—')}")
        typer.echo(f"    Commit:  {git.get('commit_hash','')[:12]}  {git.get('commit_message','')[:60]}")
        if git.get("origin_url"):
            typer.echo(f"    Origin:  {git.get('origin_url')}")

    if is_remote:
        typer.echo(f"\n  Cached at: {stats.get('local_path','')}")

    typer.echo(f"\n  Run 'openlcm scan status' to see the full index.")
    typer.echo(f"  Run 'openlcm scan export output.lcmgraph' to export for sharing.\n")


@scan_app.command("status")
def scan_status(
    repo_id: str = typer.Option("", "--repo-id", "-r", help="Repo identifier (default: show all repos)"),
    db: str = _db_option(),
):
    """Show LST graph stats. Without --repo-id shows all scanned repos."""
    from openlcm.code.graph import LSTGraph
    import datetime

    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo(f"No database at {db_path}. Run 'openlcm scan repo <path>' first.")
        raise typer.Exit(0)

    graph = LSTGraph(str(db_path))

    if not repo_id:
        # Show all repos
        repos = graph.list_repos()
        if not repos:
            typer.echo("No repos scanned yet. Run 'openlcm scan repo <path>'.")
            raise typer.Exit(0)
        typer.echo(f"\nOpenLCM LST — {len(repos)} repo(s) in {db_path}\n")
        for repo in repos:
            rid = repo["repo_id"]
            stats = graph.get_stats(rid)
            ts = repo.get("last_scanned", 0)
            scanned = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
            typer.echo(f"  ● {rid}")
            if repo.get("origin_url"):
                typer.echo(f"    URL:     {repo['origin_url']}")
            if repo.get("branch"):
                typer.echo(f"    Branch:  {repo['branch']}  commit: {repo.get('commit_hash','')[:12]}")
            typer.echo(f"    Files:   {stats['files']:,}  Symbols: {stats['symbols']:,}  Edges: {stats['edges']:,}")
            typer.echo(f"    Scanned: {scanned}")
            typer.echo("")
        return

    stats = graph.get_stats(repo_id, include_repo_meta=True)
    files = graph.list_files(repo_id, limit=10)

    typer.echo(f"\nOpenLCM LST Status — repo_id={repo_id}")
    typer.echo(f"  Database: {db_path}")
    typer.echo(f"  Files:    {stats['files']:,}")
    typer.echo(f"  Symbols:  {stats['symbols']:,}")
    typer.echo(f"  Edges:    {stats['edges']:,}")

    git = stats.get("git", {})
    if git:
        if git.get("origin_url"):
            typer.echo(f"  Origin:   {git['origin_url']}")
        if git.get("branch"):
            typer.echo(f"  Branch:   {git['branch']}  commit: {git.get('commit_hash','')[:12]}")

    if files:
        typer.echo(f"\n  Files (sample):")
        for f in files[:8]:
            ts = f.get("last_scanned", 0)
            scanned = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
            typer.echo(f"    {f['file_path']:<55} {scanned}")
    typer.echo("")


@scan_app.command("export")
def scan_export(
    output: str = typer.Argument(..., help="Output file path (e.g. myapp.lcmgraph)"),
    repo_id: str = typer.Option("default", "--repo-id", "-r", help="Repo to export"),
    repo_path: str = typer.Option("", "--repo-path", help="Informational repo path to embed in export"),
    db: str = _db_option(),
    no_compress: bool = typer.Option(False, "--no-compress", help="Write plain JSON instead of gzip"),
):
    """
    Export the LST graph as a portable .lcmgraph file.

    The exported file is self-contained and can be imported into any
    OpenLCM database — on another machine, by another agent, or shared
    across sessions.

    Example:
        openlcm scan export myapp.lcmgraph
        openlcm scan export myapp.lcmgraph --repo-id myapp
    """
    from openlcm.code.graph import LSTGraph

    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo(f"No database at {db_path}. Run 'openlcm scan repo <path>' first.")
        raise typer.Exit(1)

    graph = LSTGraph(str(db_path))
    result = graph.export_graph(
        output,
        repo_id=repo_id,
        repo_path=repo_path,
        compress=not no_compress,
    )

    if "error" in result:
        typer.echo(f"Error: {result['error']}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nExported LST graph:")
    typer.echo(f"  File:    {result['path']}")
    typer.echo(f"  Size:    {result['size_kb']:.1f} KB {'(gzipped)' if not no_compress else '(plain JSON)'}")
    typer.echo(f"  Files:   {result['files']:,}")
    typer.echo(f"  Symbols: {result['symbols']:,}")
    typer.echo(f"  Edges:   {result['edges']:,}")
    typer.echo(f"\n  Share this file with any agent or import into another session:")
    typer.echo(f"  openlcm scan import {result['path']}\n")


@scan_app.command("import")
def scan_import(
    input_file: str = typer.Argument(..., help="Path to the .lcmgraph file to import"),
    repo_id: str = typer.Option("", "--repo-id", "-r", help="Override repo_id from file"),
    db: str = _db_option(),
):
    """
    Import a portable .lcmgraph file into the local LST database.

    After import, all lcm_lst_* agent tools can query the imported graph.
    Idempotent — safe to re-import the same file.

    Example:
        openlcm scan import myapp.lcmgraph
        openlcm scan import myapp.lcmgraph --repo-id myapp --db /tmp/test.db
    """
    from openlcm.code.graph import LSTGraph

    input_path = Path(input_file).expanduser().resolve()
    if not input_path.exists():
        typer.echo(f"File not found: {input_path}", err=True)
        raise typer.Exit(1)

    db_path = _resolve_db(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    graph = LSTGraph(str(db_path))
    result = graph.import_graph(str(input_path), target_repo_id=repo_id or None)

    if "error" in result:
        typer.echo(f"Error: {result['error']}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nImported LST graph:")
    typer.echo(f"  Database: {db_path}")
    typer.echo(f"  Repo ID:  {result['repo_id']}")
    typer.echo(f"  Files:    {result['files']:,}")
    typer.echo(f"  Symbols:  {result['symbols']:,}")
    typer.echo(f"  Edges:    {result['edges']:,}")
    typer.echo(f"  Exported: {result.get('exported_at','—')}")
    typer.echo(f"\n  Agents can now use lcm_lst_* tools against this graph.\n")


@scan_app.command("visualize")
def scan_visualize(
    repo_id: str = typer.Option("", "--repo-id", "-r", help="Repo to visualize (default: auto-detect)"),
    db: str = _db_option(),
    output: str = typer.Option("", "--output", "-o", help="HTML output path (default: <repo_id>_graph.html)"),
    terminal: bool = typer.Option(False, "--terminal", "-t", help="Show rich tree in terminal instead of generating HTML"),
    max_symbols: int = typer.Option(2000, "--max-symbols", help="Max symbols to include in graph"),
    max_edges: int = typer.Option(5000, "--max-edges", help="Max edges to include in graph"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open HTML in browser after generating"),
):
    """
    Visualize the scanned repository as an interactive graph.

    Terminal mode shows a rich file/symbol tree.
    Default mode generates a self-contained HTML file with a D3.js force-directed
    graph — all nodes, edges, and metadata; no AI-generated markup.

        openlcm scan visualize                        # HTML, opens browser
        openlcm scan visualize --terminal             # Rich tree in terminal
        openlcm scan visualize --output myapp.html    # HTML to specific path
        openlcm scan visualize --repo-id myapp        # specific repo
    """
    from openlcm.code.graph import LSTGraph
    from openlcm.code.visualize import build_graph_data, render_html, render_terminal

    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo(f"No database at {db_path}. Run 'openlcm scan repo <path>' first.")
        raise typer.Exit(1)

    graph = LSTGraph(str(db_path))

    # Auto-detect repo_id if not provided
    if not repo_id:
        repos = graph.list_repos()
        if not repos:
            typer.echo("No repos scanned yet. Run 'openlcm scan repo <path>' first.")
            raise typer.Exit(1)
        if len(repos) == 1:
            repo_id = repos[0]["repo_id"]
        else:
            typer.echo(f"Multiple repos found — pick one with --repo-id:\n")
            for r in repos:
                typer.echo(f"  {r['repo_id']}")
            typer.echo("")
            raise typer.Exit(1)

    stats = graph.get_stats(repo_id)
    if stats["files"] == 0:
        repos = graph.list_repos()
        hint = f"  Available: {', '.join(r['repo_id'] for r in repos)}" if repos else ""
        typer.echo(f"No data for repo_id='{repo_id}'.{hint}")
        raise typer.Exit(1)

    if terminal:
        render_terminal(graph, repo_id)
        return

    # HTML output
    out_path = output or f"{repo_id.replace('/', '-').replace('--', '_')}_graph.html"
    typer.echo(f"\nBuilding graph data for '{repo_id}'…")

    data = build_graph_data(graph, repo_id, max_symbols=max_symbols, max_edges=max_edges)

    if data.get("truncated"):
        typer.echo(
            f"  [!] Large repo — showing {len(data['nodes'])} nodes / {len(data['edges'])} edges "
            f"(use --max-symbols / --max-edges to adjust)"
        )

    render_html(data, out_path)
    size_kb = Path(out_path).stat().st_size / 1024

    typer.echo(f"  Nodes:  {len(data['nodes']):,}  ({stats['symbols']:,} total symbols)")
    typer.echo(f"  Edges:  {len(data['edges']):,}  ({stats['edges']:,} total edges)")
    typer.echo(f"  Output: {Path(out_path).resolve()}  ({size_kb:.0f} KB)\n")

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{Path(out_path).resolve()}")



def main():
    app()


if __name__ == "__main__":
    main()
