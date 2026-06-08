"""OpenLCM Lossless Semantic Tree (LST) — codebase graph for agents.

Parses a repository once (AST → semantic graph) and stores the result in the
same SQLite database used by the rest of OpenLCM.  Agents query the graph
instead of reading files, so the entire repo never needs to live in context.

Quick start::

    from openlcm.code import LSTGraph, RepoScanner

    graph   = LSTGraph("myapp.db")
    scanner = RepoScanner()
    stats   = scanner.scan("/path/to/repo", graph)
    # {"scanned": 42, "skipped": 0, "symbols": 1823, "elapsed_s": 1.4}

    results = graph.find_symbol("PaymentService")
    cls     = graph.get_class("PaymentService")
"""

from .graph import LSTGraph
from .scanner import RepoScanner
from .git import GitCloner, is_url, get_repo_metadata

__all__ = ["LSTGraph", "RepoScanner", "GitCloner", "is_url", "get_repo_metadata"]
