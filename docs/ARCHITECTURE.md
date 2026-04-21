# Architecture

> Technical map. Grows as the package is built. For the design contract and KPI
> catalog, see `CLAUDE.md` "Project-Specific Context".

## Overview

`agent-memory-benchmark` evaluates agent memory systems against three benchmark
datasets (LongMemEval, LOCOMO, BEAM) and produces reproducible scorecards.
The benchmark is a **measurement instrument**: it calls *into* memory systems
through a transport-neutral adapter layer and does not depend on any specific
memory system.

## System diagram

```
          ┌───────────────────────────────────────────────────────┐
          │  Runner                                               │
          │  (ingest → answer → judge, cache-aware, replicate-  ) │
          │   driven)                                             │
          │      │                                                │
          │      ▼                                                │
          │  MemoryAdapter (ABC)                                  │
          │    ├─ PythonAdapter  ─▶ imported class, in-process    │
          │    ├─ HttpAdapter    ─▶ REST service (openapi.yaml)   │
          │    └─ FullContext    ─▶ null baseline (in-repo)       │
          └───────────────────────────────────────────────────────┘
                           │
                           ▼
                 Memory system's own API
               (no dependency on this repo)

  Datasets           LLM providers          Cache             Scorecard
  ├─ longmemeval     ├─ ollama              ├─ ingestion     ├─ quality
  ├─ locomo          └─ openai              ├─ answer        ├─ wall-time
  └─ beam                                   └─ judge         ├─ footprint
                                                             ├─ evidence
                                                             └─ throughput
```

## Key components

| Component | Purpose | Location |
|-----------|---------|----------|
| `compat.MemorySystemShape` | Published `typing.Protocol` spec — documentation, not dependency | `src/agent_memory_benchmark/compat.py` |
| `compat/<system>_shim.py` | Reference mapper functions (`session_mapper` / `result_mapper`) for specific memory systems whose types diverge from the benchmark's shape | `src/agent_memory_benchmark/compat/` |
| `adapters/` | Transport-layer abstraction: Python, HTTP, full-context baseline | `src/agent_memory_benchmark/adapters/` |
| `datasets/` | Dataset loaders producing uniform `BenchmarkCase` iterators | `src/agent_memory_benchmark/datasets/` |
| `llm/` | Ollama + OpenAI providers, unified `LLMProvider` protocol | `src/agent_memory_benchmark/llm/` |
| `judge/` | Dataset-specific judge prompts with byte-stable fingerprints | `src/agent_memory_benchmark/judge/` |
| `cache/` | Three-layer cache (ingestion / answer / judge), SHA-256 keyed, M3 publish guard | `src/agent_memory_benchmark/cache/` |
| `runner/` | Orchestrator, manifest, replicate driver, results directory management | `src/agent_memory_benchmark/runner/` |
| `results/` | Scorecard aggregation + JSON/Markdown/rich rendering | `src/agent_memory_benchmark/results/` |
| `cli/` | `amb` / `agent-memory-benchmark` subcommands | `src/agent_memory_benchmark/cli/` |

## Data flow

1. **Load dataset** — `load_dataset("longmemeval", split="s", ...)` yields `BenchmarkCase` objects (conversation + QAs + evidence annotations).
2. **Resolve adapter** — `resolve_adapter("python:pkg.mod:Class", memory_config=...)` constructs a `MemoryAdapter`.
3. **Ingest** — for each case, each session is passed to `adapter.ingest_session(session, case_id)` in chronological order. Wall time is measured by the runner.
4. **Answer** — for each QA, `adapter.answer_question(question, case_id)` returns an `AnswerResult`. Cache is consulted first; misses produce new answers.
5. **Judge** — `JudgeClient.judge(benchmark, question, gold, generated, ...)` produces a pass/fail (LongMemEval, BEAM) or majority vote over N runs (LOCOMO). Cache is consulted first.
6. **Score** — `scorecard.build(records, dataset)` emits JSON, Markdown, and a rich-console render.

## Cache layout

```
cache/
├── ingestion/<memory_system_id>/<key>/state.json   # state_hash, saved via save_state
├── answers/<key>.json                               # answer + timings + retrieved units
├── judge/<key>.json                                 # label + per-run raw responses
└── cache_index.json                                 # metadata (keys, mtimes, ttls)
```

Keys are SHA-256 over `\x1e`-joined inputs; exact inputs are documented in
`src/agent_memory_benchmark/cache/keys.py` and locked by regression tests.

## Results layout

```
results/<timestamp>_<benchmark>_<memory>_<answer_llm>/
├── answers.json     # manifest + per-question QARecord
├── scorecard.json   # machine-readable metrics
├── scorecard.md     # human-readable render
└── meta.json        # reproducibility manifest (git sha, revisions, argv, ...)

results/latest       # symlink / Windows junction to most recent run
```

## Non-functional invariants

- **Bit-stable prompts.** Judge prompt text is a module constant with a SHA-256 fingerprint checked in CI (`tests/unit/test_judge_prompts_stable.py`). Changes require an explicit re-baseline and are called out in `docs/methodology.md`.
- **Byte-stable cache keys.** Key inputs and separators preserve compatibility with the predecessor repo so scorecards remain comparable.
- **Independent timing.** The runner measures wall time for every phase. Adapter self-reports (in `AnswerResult.retrieval_time_ms`, etc.) are stored alongside but not substituted.
- **Pinned dataset revisions.** HF loaders call `load_dataset(..., revision=<sha>)`; the revision SHA is recorded in `meta.json`. No floating-`main` loads.
- **Strict typing.** `mypy --strict` in CI; `py.typed` marker published in the wheel.

## Key decisions

| Decision | Rationale | Date |
|----------|-----------|------|
| Benchmark calls memory systems, not vice versa | Memory systems must remain independent; benchmark is measurement, not integration point | 2026-04-20 |
| Adapter-based transport (Python, HTTP, FullContext) | Supports any memory system API shape (in-process, service, null baseline) | 2026-04-20 |
| `--judge-model` has no default | Enforces conscious opt-in for paid-API judges | 2026-04-20 |
| Whitespace + SQuAD-normalization as default evidence tokenizer | Reproducible, zero-dep, dataset-neutral; tiktoken available via opt-in | 2026-04-20 |
| Runner-measured + adapter-reported timings both stored | Discrepancy is a telemetry-drift signal | 2026-04-20 |

## External dependencies

| Service | Purpose | Notes |
|---------|---------|-------|
| Ollama | Local answer + judge LLM provider | Default for zero-cost runs; digest resolved into cache key |
| OpenAI API | Optional paid answer + judge LLM provider | Behind `[openai]` extra; must be explicitly flagged |
| HuggingFace Hub | Dataset hosting for LongMemEval and BEAM | Pinned revision per dataset; SHA stored in `meta.json` |

## Local development

```
pip install -e ".[dev,openai,tiktoken]"
pytest
ruff check src tests
mypy src
```

Ollama must be running locally for integration tests that use it:
`ollama pull llama3.1:8b` (or another model referenced by the test).

## Extensibility

- **New LLM provider** — add `src/agent_memory_benchmark/llm/<name>.py` implementing `LLMProvider`; register in `parse_spec`.
- **New dataset** — add `src/agent_memory_benchmark/datasets/<name>.py` subclassing `DatasetAdapter`; add judge prompts in `judge/<name>.py` with a new fingerprint; register in `load_dataset`.
- **New transport** — add `src/agent_memory_benchmark/adapters/<name>_adapter.py` subclassing `MemoryAdapter`; register in `resolve_adapter`.
- **New mapper for an existing memory system** — add `src/agent_memory_benchmark/compat/<system>_shim.py` exporting `session_mapper` / `result_mapper` functions; reference from the CLI as `--session-mapper agent_memory_benchmark.compat.<system>_shim:to_<system>_session`.

## Reference integrations

| System | Status | Notes |
|--------|--------|-------|
| `memory.system.MultiLayerMemory` (engram) | Shim in PR-7.5 (`compat/engram_shim.py`) | Requires engram-side patch adding `memory_system_id` / `memory_version` class attrs; evidence KPIs await engram populating `source_turn_ids` on retrieved units. |

## Related

- Approved plan: `.claude/plans/let-s-rebuild-the-benchmark-humming-pretzel.md` (agent-side)
- Predecessor (reference only): `~/code/agent-memory/benchmark/`
- Engram (primary consumer): `~/code/engram/`
