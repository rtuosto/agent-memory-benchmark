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

Binds to `127.0.0.1` by default — no auth, local tool.

To view from a phone or another device on your [Tailscale](https://tailscale.com/) tailnet:

```bash
amb serve --tailscale
# amb serve → http://<your-host>.ts.net:8000  (bind=100.x.y.z, ...)
```

`--tailscale` resolves this machine's tailnet IPv4 via the `tailscale` CLI and binds to it. Tailscale's ACLs already gate access to tailnet members, so the dashboard stays off the public internet. For a custom bind (e.g. `0.0.0.0`) use `--host` instead — it's mutually exclusive with `--tailscale`.

What the dashboard does:

- **Runs list** at `/runs` — every run under `results/` plus nested runs one level deep (e.g. `results/<container>/<timestamp>_.../`). Click any column header to sort; missing values always trail. Timestamps render in the viewer's local timezone.
- **Run detail** at `/runs/{id}` — KPI cards (overall / macro accuracy, throughput), a dedicated Ingestion section (total time + per-case mean/p95 + sessions/sec), and four Chart.js visualizations (per-category accuracy, per-query latency log scale, retrieval footprint log scale, evidence KPIs).
- **Inline baseline comparison** — every run detail auto-picks the highest-accuracy run for the same benchmark as a baseline and renders deltas on every KPI card + overlays the baseline on every chart (warm palette = this run, cool palette = baseline). Dropdown at the top lets you pick a different baseline or disable comparison (`?baseline=none`).
- **Compare table at the bottom** — the same diff `amb compare` prints on the CLI, rendered as HTML so it's readable in the browser.
- **Jobs** at `/jobs` — launch new benchmark runs from a form and watch them progress. Memory adapter + model fields are dropdowns: models come from `ollama list` (cached 60s) plus a short OpenAI benchmark whitelist, and custom user entries round-trip through re-renders. Any `openai:<model>` triggers a cost-estimate confirm page (±30% band, per-role breakdown) before the job launches. State is persisted under `jobs/<id>/` (`job.json` + `stdout.log` + `stderr.log`), so history survives server restarts. Concurrency cap defaults to 1 (`--max-concurrent`); extra submissions queue and promote FIFO.

No DB — the filesystem under `results/` is the source of truth. Reloads are cheap thanks to an mtime-keyed cache.

Service authors who want to be benchmarkable over the network implement the
four required endpoints documented in [`docs/http-api.md`](docs/http-api.md)
(schema: [`openapi.yaml`](openapi.yaml)).

## Status

Pre-release. See `docs/` for the architectural and methodological contract, and the [plan](https://github.com/rtuosto/agent-memory-benchmark) for the PR-by-PR bootstrap sequence.

## License

- Code: [MIT](LICENSE)
- Dataset licenses are per-dataset (e.g., BEAM data is CC BY-SA 4.0). See `docs/datasets.md` for provenance and attribution.
