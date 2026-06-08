"""Shared utilities for all OpenLCM benchmark scripts.

Scoring, LLM call wrapper, result formatting — used by both locomo.py and longmemeval.py.
"""

from __future__ import annotations

import json
import re
import string
import time
from pathlib import Path
from typing import Any


# ── Scoring ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = sum(p_toks.count(t) for t in common) / len(p_toks)
    rec  = sum(g_toks.count(t) for t in common) / len(g_toks)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def rouge_l(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return 0.0
    m, n = len(p), len(g)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if p[i-1] == g[j-1] else max(prev[j], curr[j-1])
        prev = curr
    lcs = prev[n]
    prec = lcs / m
    rec  = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def score(pred: str, refs: list[str]) -> dict[str, float]:
    """Best score across all references."""
    if not refs:
        return {"em": 0.0, "f1": 0.0, "rl": 0.0}
    results = [{"em": exact_match(pred, r), "f1": token_f1(pred, r), "rl": rouge_l(pred, r)} for r in refs]
    return max(results, key=lambda x: x["f1"])


def mean(scores: list[dict]) -> dict[str, float]:
    if not scores:
        return {"em": 0.0, "f1": 0.0, "rl": 0.0}
    keys = list(scores[0].keys())
    return {k: sum(s.get(k, 0.0) for s in scores) / len(scores) for k in keys}


# ── LLM call ──────────────────────────────────────────────────────────────────

def make_llm(model: str, api_key: str = ""):
    """Return a synchronous (messages) -> str function using LiteLLM."""
    import litellm

    def call(messages: list[dict], system: str = "") -> str:
        full = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs: dict = dict(model=model, messages=full, temperature=0.0, max_tokens=200)
        if api_key:
            kwargs["api_key"] = api_key
        try:
            resp = litellm.completion(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  [LLM error] {e}")
            return ""

    return call


# ── OpenLCM context builder ───────────────────────────────────────────────────

def build_openlcm_context(engine, messages: list[dict], session_id: str) -> list[dict]:
    """Run messages through LCM synchronous compress. Returns compressed message list."""
    import asyncio
    engine.bind_session(session_id, context_length=100_000)
    try:
        return asyncio.get_event_loop().run_until_complete(engine.compress(messages))
    except RuntimeError:
        # No event loop running
        return asyncio.run(engine.compress(messages))


def truncate_context(messages: list[dict], char_budget: int = 16_000) -> list[dict]:
    """Naive baseline: keep the most recent messages within the char budget."""
    kept, total = [], 0
    for m in reversed(messages):
        total += len(m.get("content") or "")
        if total > char_budget:
            break
        kept.insert(0, m)
    return kept or messages[-20:]


# ── Result table ──────────────────────────────────────────────────────────────

def print_table(results: dict[str, dict], title: str = ""):
    W = 60
    conds = list(results.keys())
    if title:
        print(f"\n{'─' * W}")
        print(f"  {title}")
        print(f"{'─' * W}")

    col_w = 14
    header = f"  {'Metric':<18}" + "".join(f"{c:>{col_w}}" for c in conds)
    print(header)
    print("  " + "─" * (18 + col_w * len(conds)))

    for metric, label in [("em", "Exact Match"), ("f1", "Token F1"), ("rl", "ROUGE-L")]:
        row = f"  {label:<18}"
        for i, c in enumerate(conds):
            val = results[c].get(metric, 0.0)
            if i > 0:
                base = results[conds[0]].get(metric, 0.0)
                diff = val - base
                sign = "+" if diff >= 0 else ""
                row += f"{val:.4f} ({sign}{diff:.3f})".rjust(col_w)
            else:
                row += f"{val:.4f}".rjust(col_w)
        print(row)

    print()
    for c in conds:
        n = results[c].get("n", 0)
        t = results[c].get("elapsed_s", 0)
        print(f"  {c:<20}  n={n}  elapsed={t:.0f}s")
    print()


def print_by_type(results: dict[str, dict], title: str = ""):
    conds = list(results.keys())
    all_types: set[str] = set()
    for v in results.values():
        all_types.update(v.get("by_type", {}).keys())
    if not all_types:
        return

    print(f"\n  {title} — breakdown by question type")
    print("  " + "─" * 60)
    col_w = 16
    hdr = f"  {'Type':<24}" + "".join(f"{'F1@'+c:>{col_w}}" for c in conds)
    print(hdr)
    print("  " + "─" * (24 + col_w * len(conds)))
    for qtype in sorted(all_types):
        row = f"  {qtype:<24}"
        for c in conds:
            f1 = results[c].get("by_type", {}).get(qtype, {}).get("f1", 0.0)
            row += f"{f1:.4f}".rjust(col_w)
        print(row)
    print()


def save_results(results: dict, benchmark: str, out_dir: str = ".") -> Path:
    out = Path(out_dir) / f"{benchmark}_results_{int(time.time())}.json"
    out.write_text(json.dumps({"benchmark": benchmark, "results": results}, indent=2))
    print(f"  Results saved → {out}\n")
    return out
