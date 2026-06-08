#!/usr/bin/env python3
"""LoCoMo benchmark — single-session long-context retention.

Tests whether OpenLCM's DAG compression preserves facts from early in a
long conversation vs. the naive truncation baseline.

Paper:   https://arxiv.org/abs/2402.13605
Dataset: https://huggingface.co/datasets/ivanmontero/locomo

Conditions:
  truncate  — keep most-recent 16k chars (naive baseline, no memory)
  openlcm   — LCM DAG compression (what we're testing)

Usage:
    pip install -r benchmarks/requirements.txt
    pip install openlcm

    python benchmarks/run_locomo.py \\
        --model anthropic/claude-haiku-4-5-20251001 \\
        --limit 50

    # OpenAI
    python benchmarks/run_locomo.py \\
        --model openai/gpt-4o-mini \\
        --limit 50

    # Only one condition
    python benchmarks/run_locomo.py --model ... --conditions openlcm
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
    print_table, save_results,
)

DATASET_ID = "ivanmontero/locomo"

QA_SYSTEM = (
    "You are answering questions about a conversation. "
    "Answer concisely — one word, a short phrase, or one sentence. "
    "If the answer is not in the conversation, say: unknown"
)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load(limit: int, split: str = "test") -> list[dict]:
    from datasets import load_dataset
    print(f"  Loading LoCoMo ({split}, limit={limit or 'all'}) ...", flush=True)
    ds = load_dataset(DATASET_ID, split=split, trust_remote_code=True)
    items = list(ds)
    return items[:limit] if limit else items


def to_messages(item: dict) -> list[dict]:
    turns = item.get("conversation") or []
    speakers = list(dict.fromkeys(t.get("speaker", "") for t in turns if t.get("speaker")))
    role_map = {}
    if speakers:
        role_map[speakers[0]] = "user"
        if len(speakers) > 1:
            role_map[speakers[1]] = "assistant"

    msgs = []
    for t in turns:
        text = t.get("text") or t.get("utterance") or t.get("content") or ""
        if text:
            msgs.append({"role": role_map.get(t.get("speaker", ""), "user"), "content": text})
    return msgs


def qa_pairs(item: dict) -> list[dict]:
    out = []
    for p in (item.get("qa_pairs") or item.get("questions") or []):
        q = p.get("question") or p.get("q") or ""
        a = p.get("answer") or p.get("a") or p.get("answers") or ""
        refs = a if isinstance(a, list) else ([a] if a else [])
        if q and refs:
            out.append({"question": q, "refs": [str(r) for r in refs]})
    return out


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run(model: str, api_key: str, limit: int, conditions: list[str], verbose: bool):
    items = load(limit)
    llm   = make_llm(model, api_key)

    engine = None
    if "openlcm" in conditions:
        from openlcm import LCMEngine
        engine = LCMEngine(model=model)
        print("  OpenLCM engine initialized.\n")

    results: dict[str, list[dict]] = {c: [] for c in conditions}
    t0 = time.time()

    for i, item in enumerate(items):
        msgs = to_messages(item)
        pairs = qa_pairs(item)
        if not msgs or not pairs:
            continue

        if verbose:
            print(f"  [{i+1}/{len(items)}] {len(msgs)} turns, {len(pairs)} QA", flush=True)

        contexts: dict[str, list[dict]] = {}

        if "truncate" in conditions:
            contexts["truncate"] = truncate_context(msgs)

        if "openlcm" in conditions:
            sid = f"locomo-{uuid.uuid4().hex[:8]}"
            contexts["openlcm"] = build_openlcm_context(engine, msgs, sid)

        for qa in pairs:
            q = qa["question"]
            refs = qa["refs"]
            for cond, ctx in contexts.items():
                prompt = ctx + [{"role": "user", "content": q}]
                pred = llm(prompt, system=QA_SYSTEM)
                results[cond].append(score(pred, refs))

    elapsed = time.time() - t0
    return {
        cond: {**mean(scores), "n": len(scores), "elapsed_s": round(elapsed, 1)}
        for cond, scores in results.items()
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LoCoMo benchmark for OpenLCM")
    p.add_argument("--model",      required=True, help="LiteLLM model string")
    p.add_argument("--api-key",    default="",    help="API key (or set env var)")
    p.add_argument("--limit",      type=int, default=50, help="Max items (0=all)")
    p.add_argument("--conditions", default="truncate,openlcm", help="Comma-separated conditions")
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--no-save",    action="store_true", help="Don't write results JSON")
    p.add_argument("--out-dir",    default=".", help="Directory for results JSON")
    args = p.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    print("\nLoCoMo — Long-Context Retention Benchmark")
    print(f"  Model:      {args.model}")
    print(f"  Conditions: {', '.join(conds)}")
    print(f"  Limit:      {args.limit or 'all'}\n")

    results = run(
        model=args.model,
        api_key=args.api_key,
        limit=args.limit,
        conditions=conds,
        verbose=args.verbose,
    )

    print_table(results, title="LoCoMo Results")

    if not args.no_save:
        save_results(results, "locomo", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
