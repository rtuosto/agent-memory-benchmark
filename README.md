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
```

`amb --help` enumerates the subcommands (`run`, `baseline`, `rejudge`, `compare`, `summarize`, `cache`). BEAM lands in PR-11.

## Status

Pre-release. See `docs/` for the architectural and methodological contract, and the [plan](https://github.com/rtuosto/agent-memory-benchmark) for the PR-by-PR bootstrap sequence.

## License

- Code: [MIT](LICENSE)
- Dataset licenses are per-dataset (e.g., BEAM data is CC BY-SA 4.0). See `docs/datasets.md` for provenance and attribution.
