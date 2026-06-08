#!/usr/bin/env python3
"""Run all benchmarks and print a combined summary table.

Usage:
    python benchmarks/run_all.py \\
        --model anthropic/claude-haiku-4-5-20251001 \\
        --limit 50

    python benchmarks/run_all.py \\
        --model openai/gpt-4o-mini \\
        --limit 30 \\
        --out-dir results/
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import run_locomo
import run_longmemeval
from shared import print_table, print_by_type


def main():
    p = argparse.ArgumentParser(description="Run all OpenLCM benchmarks")
    p.add_argument("--model",      required=True)
    p.add_argument("--api-key",    default="")
    p.add_argument("--limit",      type=int, default=50)
    p.add_argument("--conditions", default="truncate,openlcm")
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--out-dir",    default=".")
    args = p.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    print("\n" + "═" * 60)
    print("  OpenLCM Benchmark Suite")
    print(f"  Model: {args.model}  |  Items: {args.limit or 'all'}")
    print("═" * 60)

    all_results = {}

    # ── LoCoMo ────────────────────────────────────────────────────────────
    print("\n[1/2] LoCoMo — single-session long-context retention")
    loco = run_locomo.run(
        model=args.model, api_key=args.api_key,
        limit=args.limit, conditions=conds, verbose=args.verbose,
    )
    all_results["locomo"] = loco
    print_table(loco, title="LoCoMo")

    # ── LongMemEval ───────────────────────────────────────────────────────
    print("[2/2] LongMemEval — multi-session cross-session memory")
    lme = run_longmemeval.run(
        model=args.model, api_key=args.api_key,
        limit=args.limit, variant="s", conditions=conds, verbose=args.verbose,
    )
    all_results["longmemeval"] = lme
    print_table(lme, title="LongMemEval")
    print_by_type(lme, title="LongMemEval")

    # ── Combined summary ──────────────────────────────────────────────────
    print("═" * 60)
    print("  SUMMARY — Token F1 improvement over truncation baseline")
    print("═" * 60)
    print(f"  {'Benchmark':<22} {'Truncate F1':>14} {'OpenLCM F1':>14} {'Delta':>10}")
    print("  " + "─" * 62)
    for bm, res in all_results.items():
        base_f1 = res.get("truncate", {}).get("f1", 0.0)
        lcm_f1  = res.get("openlcm",  {}).get("f1", 0.0)
        delta   = lcm_f1 - base_f1
        sign    = "+" if delta >= 0 else ""
        print(f"  {bm:<22} {base_f1:>14.4f} {lcm_f1:>14.4f} {sign+f'{delta:.4f}':>10}")
    print()

    # Save combined
    out = Path(args.out_dir) / f"all_results_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"  All results saved → {out}\n")


if __name__ == "__main__":
    main()
