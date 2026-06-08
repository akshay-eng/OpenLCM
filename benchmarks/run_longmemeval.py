#!/usr/bin/env python3
"""LongMemEval benchmark — multi-session cross-session memory.

Tests whether OpenLCM's FactStore + DAG compression enables agents to
answer questions that require remembering facts from previous sessions.

Paper:   https://arxiv.org/abs/2410.10813
Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval

Question types tested:
  single_session_user   — fact stated by the user in one session
  single_session_agent  — fact stated by the agent in one session
  cross_session         — requires connecting facts across sessions  ← biggest win
  temporal              — when did something happen
  knowledge_update      — user corrected something in a later session ← FactStore wins

Conditions:
  truncate  — concatenate all sessions, keep most-recent 16k chars
  openlcm   — feed each session through LCM separately, FactStore accumulates

Usage:
    pip install -r benchmarks/requirements.txt
    pip install openlcm

    python benchmarks/run_longmemeval.py \\
        --model anthropic/claude-haiku-4-5-20251001 \\
        --limit 50

    python benchmarks/run_longmemeval.py \\
        --model openai/gpt-4o-mini \\
        --variant l \\
        --limit 30
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    build_openlcm_context, truncate_context,
    make_llm, score, mean,
    print_table, print_by_type, save_results,
)

DATASET_ID  = "xiaowu0162/longmemeval"
QTYPES = [
    "single_session_user",
    "single_session_agent",
    "cross_session",
    "temporal",
    "knowledge_update",
]

QA_SYSTEM = (
    "You are answering questions about a series of past conversations. "
    "Answer concisely — one word, a short phrase, or one sentence. "
    "If you cannot recall the answer, say: unknown"
)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load(limit: int, split: str = "test", variant: str = "s") -> list[dict]:
    from datasets import load_dataset
    print(f"  Loading LongMemEval-{variant} ({split}, limit={limit or 'all'}) ...", flush=True)
    try:
        ds = load_dataset(DATASET_ID, f"longmemeval_{variant}", split=split, trust_remote_code=True)
    except Exception:
        ds = load_dataset(DATASET_ID, split=split, trust_remote_code=True)
    items = list(ds)
    return items[:limit] if limit else items


def to_sessions(item: dict) -> list[list[dict]]:
    """Extract list-of-sessions, each a list of OpenAI-format messages."""
    sessions = []
    for session in (item.get("sessions") or item.get("conversations") or []):
        msgs = []
        for t in session.get("messages") or session.get("turns") or []:
            role    = t.get("role", "user")
            content = t.get("content") or t.get("text") or ""
            if content:
                msgs.append({"role": role, "content": content})
        if msgs:
            sessions.append(msgs)
    return sessions


def qa_pairs(item: dict) -> list[dict]:
    out = []
    # Top-level questions list
    for p in (item.get("questions") or item.get("qa_pairs") or []):
        q    = p.get("question") or p.get("q") or ""
        a    = p.get("answer") or p.get("a") or p.get("answers") or ""
        refs = a if isinstance(a, list) else ([a] if a else [])
        qtype = p.get("type") or p.get("question_type") or "unknown"
        if q and refs:
            out.append({"question": q, "refs": [str(r) for r in refs], "type": qtype})
    # Per-type nested structure
    for qtype in QTYPES:
        for p in (item.get(qtype) or []):
            q    = p.get("question") or ""
            a    = p.get("answer") or p.get("answers") or ""
            refs = a if isinstance(a, list) else ([a] if a else [])
            if q and refs:
                out.append({"question": q, "refs": [str(r) for r in refs], "type": qtype})
    return out


# ── Context builders ──────────────────────────────────────────────────────────

def build_truncate(sessions: list[list[dict]]) -> list[dict]:
    all_msgs = [m for s in sessions for m in s]
    return truncate_context(all_msgs, char_budget=16_000)


def build_openlcm(engine, sessions: list[list[dict]]) -> list[dict]:
    """Feed each session through LCM in sequence so DAG + FactStore accumulate."""
    sid = f"lme-{uuid.uuid4().hex[:8]}"
    compressed: list[dict] = []
    for session_msgs in sessions:
        compressed = build_openlcm_context(engine, session_msgs, sid)
    return compressed


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run(model: str, api_key: str, limit: int, variant: str, conditions: list[str], verbose: bool):
    items = load(limit, variant=variant)
    llm   = make_llm(model, api_key)

    engine = None
    if "openlcm" in conditions:
        from openlcm import LCMEngine
        engine = LCMEngine(model=model)
        print("  OpenLCM engine initialized.\n")

    results:      dict[str, list[dict]] = {c: [] for c in conditions}
    type_results: dict[str, dict[str, list[dict]]] = {c: {} for c in conditions}
    t0 = time.time()

    for i, item in enumerate(items):
        sessions = to_sessions(item)
        pairs    = qa_pairs(item)
        if not sessions or not pairs:
            continue

        if verbose:
            total_turns = sum(len(s) for s in sessions)
            print(f"  [{i+1}/{len(items)}] {len(sessions)} sessions ({total_turns} turns), "
                  f"{len(pairs)} QA", flush=True)

        contexts: dict[str, list[dict]] = {}
        if "truncate" in conditions:
            contexts["truncate"] = build_truncate(sessions)
        if "openlcm" in conditions:
            contexts["openlcm"] = build_openlcm(engine, sessions)

        for qa in pairs:
            q, refs, qtype = qa["question"], qa["refs"], qa["type"]
            for cond, ctx in contexts.items():
                prompt = ctx + [{"role": "user", "content": q}]
                pred   = llm(prompt, system=QA_SYSTEM)
                s      = score(pred, refs)
                s["type"] = qtype
                results[cond].append(s)
                type_results[cond].setdefault(qtype, []).append(s)

    elapsed = time.time() - t0
    out = {}
    for cond, scores in results.items():
        out[cond] = {
            **mean(scores),
            "n": len(scores),
            "elapsed_s": round(elapsed, 1),
            "by_type": {qt: mean(ts) for qt, ts in type_results[cond].items()},
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LongMemEval benchmark for OpenLCM")
    p.add_argument("--model",      required=True, help="LiteLLM model string")
    p.add_argument("--api-key",    default="",    help="API key")
    p.add_argument("--limit",      type=int, default=50, help="Max items (0=all)")
    p.add_argument("--variant",    default="s",   choices=["s", "l"],
                   help="s=500 sessions (easier), l=1000 sessions (harder)")
    p.add_argument("--conditions", default="truncate,openlcm")
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--no-save",    action="store_true")
    p.add_argument("--out-dir",    default=".")
    args = p.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    print("\nLongMemEval — Multi-Session Memory Benchmark")
    print(f"  Model:      {args.model}")
    print(f"  Variant:    longmemeval-{args.variant}")
    print(f"  Conditions: {', '.join(conds)}")
    print(f"  Limit:      {args.limit or 'all'}\n")

    results = run(
        model=args.model,
        api_key=args.api_key,
        limit=args.limit,
        variant=args.variant,
        conditions=conds,
        verbose=args.verbose,
    )

    print_table(results, title="LongMemEval Results")
    print_by_type(results, title="LongMemEval")

    if not args.no_save:
        save_results(results, "longmemeval", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
