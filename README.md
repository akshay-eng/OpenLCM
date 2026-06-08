<div align="center">
  <img src="logo_clean.png" alt="OpenLCM" width="120" />

  # OpenLCM

  **Unbounded memory. Bounded context.**

  [![PyPI version](https://img.shields.io/pypi/v/openlcm?color=4ADE80&labelColor=0C0C0C)](https://pypi.org/project/openlcm/)
  [![Python](https://img.shields.io/pypi/pyversions/openlcm?color=60A5FA&labelColor=0C0C0C)](https://pypi.org/project/openlcm/)
  [![License: MIT](https://img.shields.io/badge/license-MIT-C084FC?labelColor=0C0C0C)](LICENSE)
  [![OS](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-FBBF24?labelColor=0C0C0C)](https://pypi.org/project/openlcm/)

  [**Website**](https://akshay-eng.github.io/OpenLCM/) · [**Docs**](https://akshay-eng.github.io/OpenLCM/docs.html) · [**PyPI**](https://pypi.org/project/openlcm/)
</div>

---

OpenLCM is a framework-agnostic Python SDK that gives AI agents **permanent, lossless memory** without ever hitting the context limit. Every message is persisted verbatim in SQLite and compressed into a hierarchical DAG of summaries. Nothing is ever lost. Any past moment is recoverable.

```bash
pip install openlcm
```

Works on **macOS · Linux · Windows** — Intel, ARM, Apple Silicon. Pure Python + SQLite, no compiled extensions.

---

## What's inside

| Layer | What it does |
|---|---|
| **Message Store** | Every message written verbatim to SQLite with a stable `store_id`. FTS5-indexed. Never modified or deleted. |
| **Summary DAG** | Old messages compress into D0 leaf nodes → D1 session arcs → D2 durable history. Always recoverable. |
| **Fact Store** | Persistent key-value memory for decisions, constraints, preferences. Tagged, linked, auto-injected. |
| **LST Code Graph** | AST → semantic graph of files, classes, functions, call edges. Agents query instead of reading files. |

---

## Quick start

```python
from openlcm import LCMEngine

engine = LCMEngine(model="anthropic/claude-haiku-4-5-20251001")
engine.bind_session("my-session", context_length=200_000)

# Call before every LLM turn — compresses only when needed
messages = await engine.compress(messages)
```

Pass any [LiteLLM](https://github.com/BerriAI/litellm) model string: `openai/gpt-4o`, `gemini/gemini-2.0-flash`, `azure/gpt-4o`, `ollama/llama3`, etc.

---

## Framework adapters

All adapters included — one install, no extras needed.

<details>
<summary><b>LangGraph</b></summary>

```python
from openlcm.adapters.langgraph import LCMCheckpointer

graph = StateGraph(MyState).compile(
    checkpointer=LCMCheckpointer(engine)
)
```
</details>

<details>
<summary><b>Google ADK</b></summary>

```python
from openlcm.adapters.google_adk import LCMSessionService, lcm_compress_callback

agent = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    before_model_callback=lcm_compress_callback(engine),
)
runner = Runner(agent=agent, session_service=LCMSessionService(engine))
```
</details>

<details>
<summary><b>AutoGen</b></summary>

```python
from openlcm.adapters.autogen import LCMContext

agent = AssistantAgent("assistant",
    model_client=client,
    model_context=LCMContext(engine))
```
</details>

<details>
<summary><b>CrewAI</b></summary>

```python
from openlcm.adapters.crewai import LCMStorage

crew = Crew(memory=True,
    long_term_memory=LongTermMemory(storage=LCMStorage(engine)))
```
</details>

<details>
<summary><b>OpenAI / Anthropic / LlamaIndex / Haystack / Gemini</b></summary>

```python
from openlcm.adapters.openai     import OpenAIMessages
from openlcm.adapters.anthropic  import AnthropicMessages
from openlcm.adapters.llamaindex import LlamaIndexMessages
from openlcm.adapters.haystack   import HaystackMessages
from openlcm.adapters.gemini     import GeminiMessages

# All follow the same to_lcm() / from_lcm() interface
lcm = OpenAIMessages.to_lcm(messages)
lcm = await engine.compress(lcm)
messages = OpenAIMessages.from_lcm(lcm)
```
</details>

---

## Persistent memory (Fact Store)

```python
# Store a fact — survives every session boundary
engine.handle_tool_call("lcm_remember", {
    "key": "constraint.no_prod_push",
    "value": "Never push to production without a PR review",
    "category": "constraint",
    "tags": ["deployment"]
})

# Auto-injected into system message before each compression turn
# Also queryable explicitly:
engine.handle_tool_call("lcm_recall", {"category": "constraint"})
```

Auto-extraction: enable `LCM_EXTRACTION_TO_FACTS_ENABLED=true` to have facts automatically pulled from each DAG summary node.

---

## LST — Lossless Semantic Tree (Codebase Graph)

Parse a repository once and store it as a queryable graph. Agents navigate structure through tools instead of reading raw files — eliminating the "re-discovery" problem where agents waste thousands of tokens re-reading the same files every session.

### Scan

```bash
# Local path
openlcm scan repo /path/to/myapp

# GitHub URL — auto-clones to ~/.openlcm/repos/
openlcm scan repo https://github.com/fastapi/fastapi

# Check what was indexed
openlcm scan status
```

```python
from openlcm.code.graph import LSTGraph
from openlcm.code.scanner import RepoScanner

graph   = LSTGraph("myapp.db")
scanner = RepoScanner()
scanner.scan("/path/to/repo", graph, repo_id="myapp")

# Or attach to engine for full tool integration
engine.attach_lst(graph, repo_id="myapp")
```

### Session context — zero re-discovery

```python
engine.bind_session(session_id, context_length=200_000)

# ~500-token orientation block: key classes, entry points, recent session history
ctx = engine.get_lst_context()

# Smart file read — full content first time, compact LST view on repeats (10× fewer tokens)
result = engine.get_file_context("payments/service.py")
# result["mode"] == "full"  (first read)
# result["mode"] == "compact"  (every repeat read this session)
```

### Pin discoveries to symbols

```python
# Pin what you found to a symbol — survives across sessions
engine.handle_tool_call("lcm_remember", {
    "key": "payments.charge.ratelimit",
    "value": "Stripe hits 100 req/s — needs exponential backoff",
    "symbol": "PaymentService.charge",   # ← linked to the symbol
    "category": "constraint"
})

# Next session — surfaces automatically when class is queried
result = engine.handle_tool_call("lcm_lst_class", {"class_name": "PaymentService"})
# result["pinned_facts"] = [{"constraint": "Stripe hits 100 req/s..."}]
```

### LST agent tools (13 tools)

| Tool | Purpose |
|---|---|
| `lcm_lst_context` | Full repo orientation in ~500 tokens — call first at session start |
| `lcm_read_file` | Smart read: full content first time, compact LST view on repeats |
| `lcm_lst_find` | FTS5 search across all symbols, docstrings, signatures |
| `lcm_lst_class` | Class + all methods + signatures + pinned facts from past sessions |
| `lcm_lst_callers` | Who calls this function (call-graph inbound) |
| `lcm_lst_callees` | What this function calls (call-graph outbound) |
| `lcm_lst_refs` | All edge references to a symbol |
| `lcm_lst_path` | Shortest dependency path between two symbols |
| `lcm_lst_ancestors` | All symbols that transitively call a function |
| `lcm_lst_descendants` | Full dependency footprint of a function |
| `lcm_lst_facts` | All agent discoveries pinned to a symbol |
| `lcm_lst_file` | All symbols in a file |
| `lcm_lst_scan` | Scan or re-scan a repository |

### Multi-language support

Python uses the stdlib `ast` module (docstrings + call edges + full signatures). All other languages use [Universal Ctags](https://ctags.io):

```bash
brew install universal-ctags     # macOS
sudo apt install universal-ctags # Ubuntu
scoop install universal-ctags    # Windows
```

Supported: TypeScript, JavaScript, Go, Java, Rust, Ruby, PHP, Swift, Kotlin, C/C++, C#, Scala, and 90+ more.

### Interactive code graph

```bash
# Generate D3.js force-directed HTML graph — opens in browser
openlcm scan visualize

# Rich tree view in terminal
openlcm scan visualize --terminal

# Export portable graph for sharing between agents / machines
openlcm scan export myapp.lcmgraph
openlcm scan import myapp.lcmgraph
```

---

## Configuration

```python
from openlcm.core.config import LCMConfig

config = LCMConfig(
    context_threshold = 0.75,   # compress at 75% of context window
    fresh_tail_count  = 64,     # protect last 64 messages from compression
    leaf_chunk_tokens = 20_000, # tokens per D0 leaf summary
    # Memory
    auto_inject_memory = True,  # auto-inject relevant facts before each compression
    auto_inject_top_k  = 5,
    extraction_to_facts_enabled = True,
    # LST
    lst_enabled   = True,
    lst_repo_path = "/path/to/repo",
    lst_repo_id   = "myapp",
    lst_auto_inject = True,     # inject repo context into every compress() call
)

engine = LCMEngine(model="...", config=config)
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `LCM_CONTEXT_THRESHOLD` | `0.75` | Compression trigger fraction |
| `LCM_FRESH_TAIL_COUNT` | `64` | Messages protected at tail |
| `LCM_LEAF_CHUNK_TOKENS` | `20000` | Tokens per D0 summary node |
| `LCM_AUTO_INJECT_MEMORY` | `false` | Auto-inject relevant facts |
| `LCM_EXTRACTION_TO_FACTS_ENABLED` | `false` | Auto-extract facts from summaries |
| `LCM_EMBEDDING_MODEL` | `""` | Enable semantic search (any LiteLLM embedding model) |
| `LCM_LST_ENABLED` | `false` | Auto-scan repo on engine init |
| `LCM_LST_REPO_PATH` | `""` | Path to scan |
| `LCM_LST_REPO_ID` | `default` | Logical repo identifier |
| `LCM_LST_AUTO_INJECT` | `true` | Inject repo context into compress() |

---

## Live dashboard

```bash
openlcm viz           # opens http://localhost:7842
openlcm grep "query"  # full-text search across all sessions
openlcm status        # session + compression stats
```

```python
import threading
from openlcm.viz.server import create_app, serve as viz_serve

threading.Thread(
    target=lambda: viz_serve(create_app(engine), port=7842, open_browser=True),
    daemon=True
).start()
```

Shows: token pressure gauge, live DAG graph, message store, fact store, and event log.

---

## Benchmarks

Standalone scripts to measure compression quality against established memory benchmarks:

```bash
pip install datasets litellm

# LoCoMo — single-session long-context retention
python benchmarks/run_locomo.py \
    --model anthropic/claude-haiku-4-5-20251001 --limit 50

# LongMemEval — multi-session cross-session memory
python benchmarks/run_longmemeval.py \
    --model anthropic/claude-haiku-4-5-20251001 --limit 50

# Both + combined summary table
python benchmarks/run_all.py \
    --model anthropic/claude-haiku-4-5-20251001 --limit 50
```

| Benchmark | What it tests | OpenLCM advantage |
|---|---|---|
| [LoCoMo](https://arxiv.org/abs/2402.13605) | Single-session retention — answer questions about turn 5 when on turn 200 | DAG preserves early-turn facts that truncation drops |
| [LongMemEval](https://arxiv.org/abs/2410.10813) | Multi-session memory — cross-session and knowledge-update questions | FactStore + DAG accumulation across sessions |

---

## CLI reference

```bash
openlcm status                    # database stats
openlcm sessions                  # list all sessions
openlcm grep "query"              # FTS5 search across history
openlcm expand --node-id 12       # recover messages from a DAG node
openlcm export SESSION_ID         # export session as JSON
openlcm doctor                    # database integrity check
openlcm viz                       # live dashboard

openlcm scan repo <path|url>      # scan repo into LST graph
openlcm scan status               # show all indexed repos
openlcm scan visualize            # interactive code graph (HTML + terminal)
openlcm scan export out.lcmgraph  # export portable graph
openlcm scan import in.lcmgraph   # import portable graph
```

---

## Guarantees

- **Lossless** — every message persisted with a stable `store_id`. Recoverable even after 100 compactions across 50 sessions.
- **Deterministic** — summarization always terminates. L1 → L2 → L3 escalation with circuit breaker prevents retry storms.
- **Zero-cost** — compression fires only when the configurable threshold is exceeded. Short conversations pay zero overhead.
- **Cross-platform** — works on macOS (Intel + Apple Silicon), Linux (x86 + ARM), and Windows. Pure Python + SQLite stdlib.

---

## License

MIT — see [LICENSE](LICENSE).

Built on the LCM paper by Ehrlich & Blackman (Voltropy, 2026).
