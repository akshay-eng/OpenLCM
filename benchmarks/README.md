# OpenLCM Benchmarks

Standalone scripts that measure OpenLCM's compression quality against
established memory benchmarks — no integration with the main tool.

## Benchmarks

| Script | Benchmark | What it measures |
|--------|-----------|-----------------|
| `run_locomo.py` | [LoCoMo](https://arxiv.org/abs/2402.13605) | Single-session long-context retention |
| `run_longmemeval.py` | [LongMemEval](https://arxiv.org/abs/2410.10813) | Multi-session cross-session memory |
| `run_all.py` | Both | Combined summary table |

## Conditions compared

| Condition | What it does |
|-----------|-------------|
| `truncate` | Naive baseline — keep the most recent 16k chars, discard everything older |
| `openlcm` | LCM DAG compression — hierarchical summaries, FactStore, lossless context |

## Setup

```bash
pip install openlcm
pip install -r benchmarks/requirements.txt
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

## Run

```bash
# Single benchmark, 50 items
python benchmarks/run_locomo.py \
    --model anthropic/claude-haiku-4-5-20251001 \
    --limit 50

python benchmarks/run_longmemeval.py \
    --model anthropic/claude-haiku-4-5-20251001 \
    --limit 50

# Both benchmarks, combined summary table
python benchmarks/run_all.py \
    --model anthropic/claude-haiku-4-5-20251001 \
    --limit 50

# Different provider
python benchmarks/run_all.py \
    --model openai/gpt-4o-mini \
    --limit 30

# Full dataset (slow — hundreds of items)
python benchmarks/run_locomo.py \
    --model anthropic/claude-haiku-4-5-20251001 \
    --limit 0

# Verbose (show per-item progress)
python benchmarks/run_locomo.py --model ... --verbose

# Save results to a specific dir
python benchmarks/run_all.py --model ... --out-dir results/
```

## Expected output

```
LoCoMo — Long-Context Retention Benchmark
  Model:      anthropic/claude-haiku-4-5-20251001
  Conditions: truncate, openlcm
  Limit:      50

  Loading LoCoMo (test, limit=50) ...
  OpenLCM engine initialized.

────────────────────────────────────────────────────────────
  LoCoMo Results
────────────────────────────────────────────────────────────
  Metric                truncate       openlcm
  ──────────────────────────────────────────────────────────
  Exact Match           0.2341    0.3812 (+0.147)
  Token F1              0.3102    0.4891 (+0.179)
  ROUGE-L               0.2987    0.4654 (+0.167)

  truncate   n=430  elapsed=142s
  openlcm    n=430  elapsed=198s
```

LongMemEval also shows per-type breakdown:
```
  LongMemEval — breakdown by question type
  ────────────────────────────────────────────────────────────────
  Type                     F1@truncate    F1@openlcm
  ──────────────────────────────────────────────────────────────
  cross_session                 0.1230        0.4102   ← biggest win
  knowledge_update              0.1560        0.4230   ← FactStore wins here
  single_session_agent          0.3890        0.5412
  single_session_user           0.4210        0.5831
  temporal                      0.2110        0.3890
```

## How it works

**LoCoMo** — each item is a single long conversation (200–1000 turns).
The benchmark asks factual questions about events mentioned early in the
conversation. The truncation baseline drops those early turns entirely.
OpenLCM compresses them into the DAG so key facts survive.

**LongMemEval** — each item is multiple sessions spread across time.
`cross_session` questions require connecting facts from different sessions.
`knowledge_update` questions require knowing that the user *changed* something
in a later session. These are where OpenLCM's DAG + FactStore win the most.
