# agent-memory-benchmark

A benchmark suite for agent memory systems. Measures **quality**, **wall-time performance**, **retrieval footprint**, and **retrieval quality against evidence annotations** across three benchmark datasets:

- [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) — `s` (100 questions) and `m` (multi-GB) splits
- [LOCOMO](https://github.com/snap-stanford/locomo) — 10 conversations across categories 1–4 (category 5 excluded as unanswerable)
- [BEAM](https://huggingface.co/datasets/Mohammadta/BEAM) — *Beyond a Million Tokens* ([arxiv 2510.27246](https://arxiv.org/abs/2510.27246)), 2000 questions × 100 conversations × 10 memory abilities

## Design posture

This benchmark is a **measurement instrument**. It calls into memory systems through a transport-neutral adapter layer. Memory systems have **zero dependency** on this repo: they expose their own API (in-process Python class, HTTP service) and the benchmark connects via a configured adapter. An optional compatibility `Protocol` (`agent_memory_benchmark.compat.MemorySystemShape`) is published as a spec memory-system authors may choose to match for zero-glue benchmarking — it is a spec, not a dependency.

The benchmark enforces a **required `--judge-model` flag** (no default) so that any use of a paid-API judge is a conscious opt-in.

## Quickstart

```bash
pip install -e ".[dev,openai,tiktoken]"

# LongMemEval S split (HF-hosted, revision-pinned) via the null baseline.
amb run longmemeval \
    --memory full-context \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b \
    --split s --limit 5

# LOCOMO (local file; not redistributed) with 10-run majority-vote judging.
amb run locomo \
    --memory full-context \
    --data ./locomo10.json \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b \
    --judge-runs 10 --limit 1

# Remote memory system over HTTP — contract in openapi.yaml / docs/http-api.md.
amb run longmemeval \
    --memory http://localhost:8000 \
    --memory-header Authorization="Bearer ${TOKEN}" \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b \
    --split s --limit 5

# BEAM (Beyond a Million Tokens) — ten-ability routing, 128K-1M token contexts.
amb run beam \
    --memory full-context \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b \
    --variant beam --abilities temporal-reasoning,abstention \
    --limit 20
```

`amb --help` enumerates the subcommands (`run`, `baseline`, `rejudge`, `compare`, `summarize`, `cache`, `serve`). BEAM variant / split / ability filter semantics are documented in [`docs/beam.md`](docs/beam.md).

### Web dashboard (`amb serve`)

Optional local dashboard for browsing runs, comparing scorecards, and visualizing KPIs. Install the extra first:

```bash
pip install -e ".[web]"
amb serve --port 8000 --results-dir results
```

Binds to `127.0.0.1` only — no auth, local tool. What it does:

- **Runs list** at `/runs` — sortable table of every run under `results/`, one row per `<timestamp>_<benchmark>_<memory>_<model>_<tag>/` directory.
- **Run detail** at `/runs/{id}` — KPI cards (overall / macro accuracy, throughput), a dedicated Ingestion section (total time + per-case mean/p95 + sessions/sec), and four Chart.js visualizations (per-category accuracy, per-query latency log scale, retrieval footprint log scale, evidence KPIs).
- **Inline baseline comparison** — every run detail auto-picks the highest-accuracy run for the same benchmark as a baseline and renders deltas on every KPI card + overlays the baseline on every chart (warm palette = this run, cool palette = baseline). Dropdown at the top lets you pick a different baseline or disable comparison (`?baseline=none`).
- **Compare table at the bottom** — the same diff `amb compare` prints on the CLI, rendered as HTML so it's readable in the browser.
- **Jobs** at `/jobs` — launch new benchmark runs from a form and watch them progress. Local (Ollama) jobs submit immediately; any job with an `openai:<model>` answer or judge spec redirects to a cost-estimate confirm page first (pricing snapshot in `web/cost.py`). Job state (`queued`/`running`/`succeeded`/`failed`) is persisted under `jobs/<id>/` as `job.json` + `stdout.log` + `stderr.log`, so history survives server restarts. Concurrency cap defaults to 1 (`--max-concurrent`); extra submissions queue and promote FIFO as slots free up.

No DB — the filesystem under `results/` is the source of truth. Reloads are cheap thanks to an mtime-keyed cache.

Service authors who want to be benchmarkable over the network implement the
four required endpoints documented in [`docs/http-api.md`](docs/http-api.md)
(schema: [`openapi.yaml`](openapi.yaml)).

## Status

Pre-release. See `docs/` for the architectural and methodological contract, and the [plan](https://github.com/rtuosto/agent-memory-benchmark) for the PR-by-PR bootstrap sequence.

## License

- Code: [MIT](LICENSE)
- Dataset licenses are per-dataset (e.g., BEAM data is CC BY-SA 4.0). See `docs/datasets.md` for provenance and attribution.
