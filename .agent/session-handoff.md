# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

---

## Session: 2026-04-20 — PR-7.5 mapper CLI flags + engram wrapper shim

### What Was Done

**PR-7.5 (on `feat/runner`, stacked atop PR-7):** landed two complementary additions that together make it possible to benchmark engram end-to-end with zero engram-side changes.

- `src/agent_memory_benchmark/adapters/factory.py` — `resolve_adapter` now accepts `session_mapper` / `result_mapper` kwargs and forwards them to `PythonAdapter.from_spec`. Passing mappers with `full-context` is an explicit `AdapterSpecError` (no silent drop).
- `src/agent_memory_benchmark/runner/__init__.py` — new `_resolve_callable(spec, *, flag)` helper that resolves `"pkg.module:function"` strings via `importlib` with flag-name-aware errors (missing colon / empty sides / ImportError / missing attribute / non-callable target). `run_benchmark(...)` accepts `session_mapper_spec` / `result_mapper_spec` string kwargs and resolves them once at assembly time.
- `src/agent_memory_benchmark/cli/run_cmd.py` — new `--session-mapper pkg.mod:fn` / `--result-mapper pkg.mod:fn` CLI flags threaded through to `run_benchmark`.
- `src/agent_memory_benchmark/compat.py` → `src/agent_memory_benchmark/compat/__init__.py` via `git mv` (history preserved) so `compat/` can hold per-memory-system shims.
- `src/agent_memory_benchmark/compat/engram_shim.py` — `EngramShim` wrapper class. Declares `memory_system_id: ClassVar[str] = "engram"` on the class itself and reads `memory_version` from engram's `MULTI_LAYER_MEMORY_VERSION` module constant at `__init__` time. Holds an inner `MultiLayerMemory`, translates types at the boundary. Import of `memory.system` is *lazy* (inside `__init__`) so the shim module stays importable when engram isn't installed. `__init__(**kwargs)` forwards to `MultiLayerMemory(**kwargs)` so existing `--memory-config` plumbing works unchanged.
- `_to_engram_session` / `_from_engram_answer` mappers. Session translation tries `benchmark.datasets.locomo.Session` / `DialogueTurn` first, falls back to local duck-typed classes (`_DuckSession` / `_DuckTurn`) with `__slots__` for the required attribute names. Answer mapper reads attributes permissively (`getattr(raw, ..., default)`) so minor engram schema shifts don't break things; if engram starts populating `retrieved` with `source_turn_ids`, it flows through unchanged.

**Tests (22 new; 290 total):**
- `test_compat_engram_shim.py` — shim instantiation without engram raises a targeted `ImportError`, duck-fallback triggers when `benchmark.datasets.locomo` isn't importable, field-name mapping round-trip (including `None` image_caption), answer mapper builds `AnswerResult` from duck-typed objects, preserves `retrieved` when present, tolerates missing optional fields + coerces null timings.
- `test_runner_mapper_resolution.py` — every error path on `_resolve_callable`, and a parametrize asserting the flag name appears in every error.
- `test_adapters_factory.py` updates — mapper kwargs forward to `PythonAdapter`; full-context rejects mapper kwargs.
- `test_cli_run_cmd.py` updates — `--session-mapper` / `--result-mapper` captured on the namespace; default to `None` when omitted.

**Invocation (the payoff):**

```bash
amb run longmemeval \
    --memory python:agent_memory_benchmark.compat.engram_shim:EngramShim \
    --memory-config embedding_model=sentence-transformers/all-MiniLM-L6-v2 \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b \
    --split s --limit 5
```

### Current State

- Branch: `feat/runner` (not yet merged). Three PR-level commits (`d4500d3` PR-7 impl, `45ddb5f` handoff, `6748a2d` docs, `d1029af` docs correction, `e6099c3` PR-7.5 impl).
- Tests: 290 passed, 1 skipped (POSIX-only symlink test).
- Lint: `ruff check`, `ruff format --check` → clean.
- Types: `mypy src` → clean (36 source files).

### What's Next

- Merge chain: PR-6 → PR-7 → PR-7.5. All sit in order on `feat/runner`.
- **PR-8** — CLI subcommands unchanged in scope.

### Open Questions

- **When to ship a real engram integration test** — the shim is currently only unit-tested (engram not installed in this repo's venv). The right place for a real-engram smoke test is `tests/integration/` with a conditional skip when engram isn't on `PYTHONPATH`. Probably folds in alongside PR-13's integration-test work with recorded HTTP fixtures.
- **Evidence KPIs on engram runs** — will stay `null` until engram populates `RetrievedUnit.source_turn_ids`. That's an engram-side change, not in scope for this repo.

### Gotchas

- **Duck-typed engram session.** The shim's `_DuckSession` / `_DuckTurn` fallback relies on engram's code doing attribute access, not `isinstance(session, Session)` checks. If engram ever tightens to `isinstance`, the fallback needs to be replaced with a subclass of the real types (requires engram be importable, which restricts test coverage). The duck objects use `__slots__` to match the minimal shape exactly.
- **`_resolve_callable` runs at assembly time**, before the adapter opens. So bad mapper specs surface *before* any LLM connection is made — the error message says which flag was wrong and why.
- **Mapper function path vs. wrapper class path.** Docs spell out the distinction: wrapper class for any class-signature divergence (missing `memory_system_id`, different method names), mapper flags only when the target already matches `MemorySystemShape` structurally and just the *value types* need translating.

### How to pick up from here

```
cd ~/code/agent-memory-benchmark
source .venv/Scripts/activate
# Once PR-6 / PR-7 / PR-7.5 are all merged:
git checkout main
git checkout -b feat/cli-subcommands
# Start PR-8: baseline / rejudge / compare / summarize / cache commands.
```

---

## Session: 2026-04-20 — PR-7 runner + manifest + scorecard + `amb run`

### What Was Done

**PR-7 (on `feat/runner`, branched from `feat/longmemeval` — PR-6 is still unmerged, so PR-7 is stacked atop it and will ride in after PR-6 merges):**

- `runner/manifest.py` — `QARecord` (one row per question, with runner- + adapter-measured timings + `answer_discrepancy_ms` drift signal + `evidence_turn_ids` / `retrieved_turn_ids` for the evidence KPIs), `RunManifest` (every field needed to reproduce the run — model specs + resolved digests, dataset descriptor hash + HF revision, judge prompt fingerprint, benchmark git state, full CLI argv, protocol version 0.1), `RunDir` (typed wrapper for `answers.json` / `meta.json` / `scorecard.{json,md}`), `save_run_file` / `load_run_file` (forward-compat: unknown fields on disk are dropped).
- `runner/latest.py` — `update_latest_pointer` tries symlink → Windows junction → `latest.txt` fallback. Never hard-fails; preserves a real `latest/` directory if present.
- `runner/judge_adapter.py` — `BenchmarkJudge` Protocol + `LongMemEvalJudge` concrete class. The judge returns the *template* fingerprint (not the formatted prompt) so re-baselining a template invalidates the right cohort. LongMemEval rejects `--judge-runs > 1` at construction time since upstream protocol is single-run yes/no.
- `runner/orchestrator.py` — `BenchmarkRunner`: the ingest → answer → judge loop. Cache-aware for all three caches, resume-aware via `answers.json` keys, writes the run file after each QA so Ctrl-C is safe. `_load_cached_judge` mirrors the write path exactly so cache-hit lookups always find what was written.
- `runner/__init__.py` — `run_benchmark(...)` assembly-time entry point. Resolves git state, provider specs (Ollama digest pinning via `resolve_spec`), builds the `RunManifest`, opens/closes the adapter + both providers with `try/finally`, and calls `_finalize_artifacts` to produce `scorecard.{json,md}` + update `results/latest`.
- `results/scorecard.py` — four KPI families per plan: **quality** (overall + macro accuracy, token-F1 with SQuAD-style normalization, per-category), **wall-time perf** with `{mean, p50, p95, max}` distributions for every timing bucket + the `answer_discrepancy_ms` drift signal, **retrieval footprint** (units + tokens per query), **evidence KPIs** (turn/unit/token × completeness/density), plus the `throughput.queries_per_sec` / `throughput.sessions_per_sec` headline. `scorecard_to_dict` locks the public JSON shape (tested).
- `results/render.py` — `render_scorecard_markdown` for `scorecard.md` (throughput headline → quality → per-category → latency → retrieval footprint → evidence KPIs → methodology) + `print_scorecard_rich` for the console. `rich` import is lazy so `build_scorecard` stays pure-stdlib.
- `cli/run_cmd.py` + `cli/main.py` — `amb run <dataset> ...` with every knob from the plan. `--memory-config KEY=VALUE` parses values as JSON when possible, strings otherwise (so `timeout=30` is an int, `model=llama3` stays a string).

**Tests (60 new; 268 total; 1 skipped on Windows):**
- `test_runner_manifest.py` — save/load roundtrip, drop-unknown forward-compat, path-component sanitization (including `...` → `unnamed`), directory name composition.
- `test_runner_latest.py` — POSIX symlink success (skipped on win32), real-directory preservation, `latest.txt` fallback when symlink + junction both fail, Windows permission-denied simulation.
- `test_runner_judge_adapter.py` — task-to-template routing including abstention override, unsupported-task rejection, multi-run rejection (single-run protocol lock).
- `test_runner_orchestrator.py` — full end-to-end loop with fake providers + real `FullContextAdapter`. Covers: primary path (answer + judge produce one record), answer-cache hit skips LLM on second run, judge-cache hit skips judge on second run, `--no-cache` forces regeneration, resume skips completed QA, ingestion-state round-trip.
- `test_scorecard.py` — KPI shapes, macro vs overall divergence under size skew, SQuAD token-F1 normalization, evidence KPIs when retrieval absent vs. present, replicate stats, public JSON shape lock.
- `test_render_scorecard.py` — all required markdown sections present.
- `test_cli_run_cmd.py` — arg parsing, memory-config JSON coercion, required-arg enforcement, unknown-dataset rejection.

### Current State

- Branch: `feat/runner` (not yet merged). Head: `feat(runner,results): orchestrator + manifest + scorecard + amb run CLI`. Built on top of `feat/longmemeval` (which still needs to merge first).
- Tests: `pytest tests/unit -q` → 268 passed, 1 skipped (`test_symlink_path_on_posix` is Windows-skipped).
- Lint: `ruff check src tests`, `ruff format --check` → clean.
- Types: `mypy src` → clean (35 source files).
- CLI: `amb --version` still works; `amb run --help` lists every documented flag.

### What's Next

- Merge chain: PR-6 → PR-7 (they sit in that order on `feat/runner`).
- **PR-8** — CLI subcommands: `baseline` (shortcut for `--memory full-context`), `rejudge` (reload stored generations + re-run judge with different model/prompt fingerprint; fingerprint drift should auto-invalidate the judge cache), `compare` (diff two `scorecard.json`), `summarize` (pretty-print `answers.json`), `cache {info|clear|gc}`.
- PR-9 — LOCOMO loader + LOCOMO judge (10-run majority). Second implementer of `BenchmarkJudge`.
- PR-10 — HTTP adapter + `openapi.yaml`.
- PR-11 — BEAM loader + ability-specific judge prompts.
- PR-12 — noise-aware replicates (K1/K6) + `--publishable` gate + `docs/methodology.md`.

### Open Questions

- **Ollama digest in cache key locked.** `run_benchmark` now calls `provider.resolve_spec()` before manifest construction, so `ollama:llama3.1:8b@sha256:<digest>` flows into the answer cache key automatically.
- **Evidence-token KPIs are scaffolded but not populated.** The code path collects retrieved/evidence turn IDs and computes turn + unit completeness/density. Token-level completeness/density needs per-turn text recovery to map retrieved text back onto evidence-turn text; implementation deferred since the full-context baseline doesn't retrieve anything meaningful anyway. Revisit in PR-8 or when a real memory system (PythonAdapter against engram) lands.
- **`--publishable` gate** — not implemented this PR; reserved for PR-12 alongside replicates. M3 guard from PR-4 is still the fallback for cache-version drift.
- Still open: BEAM evidence-turn field (PR-11).

### Gotchas

- **PR-7 is stacked on PR-6.** Both branches need to merge in order (PR-6 first, then PR-7). Feature work beyond PR-7 can be branched off `feat/runner` as further stacking, or wait for both to land on main.
- **FullContextAdapter `retrieved=()` → evidence KPIs null.** The null baseline returns no retrieval units, so turn/unit/token evidence metrics are null for every question. That's correct behavior, but it means the first end-to-end scorecard will show `evidence.turn_completeness: null` etc. The non-null numbers only materialize once a memory adapter populates `RetrievedUnit.source_turn_ids`.
- **Judge cache lookup is benchmark-specific.** `BenchmarkRunner._load_cached_judge` hard-codes LongMemEval template selection because it needs the same template fingerprint the judge used for the write. When LOCOMO / BEAM land, refactor this into the `BenchmarkJudge` protocol (e.g. a `cache_fingerprint(qa)` method) so the orchestrator stays benchmark-agnostic.
- **`run_benchmark` closes `answer_provider` twice** — once via `adapter.close()` (which owns it for `FullContextAdapter`), once directly. Both ollama and openai close methods are idempotent (they `None`-out the httpx client after the first call). Documented in a code comment near the `try/finally`.
- **`amb run` catches broad exceptions at the CLI boundary** (`except Exception`) to translate to a printable error + non-zero exit code. Ruff `BLE001` is `# noqa`'d there — this is the right place for it because any failure needs to surface as a CLI error, not a stack trace.

### How to pick up from here

```
cd ~/code/agent-memory-benchmark
source .venv/Scripts/activate
# Once PR-6 and PR-7 are both merged:
git checkout main
git checkout -b feat/cli-subcommands
# Start PR-8: baseline / rejudge / compare / summarize / cache commands.
```

---

## Session: 2026-04-20 — PR-6 LongMemEval loader + judge prompts

### What Was Done

**PR-6 (on `feat/longmemeval`, branched from `main` after the PR-5 merge):**

- `src/agent_memory_benchmark/datasets/base.py` — `DatasetAdapter` ABC: `__iter__` yields `BenchmarkCase`, `__len__`, `descriptor_hash()` flows into cache keys.
- `datasets/longmemeval.py` — HF `xiaowu0162/longmemeval-cleaned` loader. HF revision pinned to `98d7416c24c778c2fee6e6f3006e7a073259d48f` (2025-09-19) via `resolve/<revision>/<filename>` URL — no floating `main` loads. `load_s()` pulls the S split over HTTP through the HF URL; `load_m_local()` reads the multi-GB M file from disk. Lazy `from datasets import load_dataset` inside the loader functions so the module imports cleanly on machines where pandas can't load (Windows App Control on this host blocks the pandas DLL). Row converter `_row_to_case` produces `BenchmarkCase` with `Turn(turn_id=f"{session_id}:{idx}")`, expands `answer_session_ids` to every turn inside those sessions as `evidence_turn_ids`, and records abstention as `metadata["abstention"]` ("1"/"0"). `_stratified_indices` ports the predecessor's proportional-allocation + deterministic round-robin selector. `descriptor_hash` = sha256 of `(name, split, revision, f"{applied_strategy}:{limit|all}")` joined with `\x1e`.
- `datasets/__init__.py` — `load_dataset(name, **kwargs)` dispatcher. LOCOMO / BEAM raise `DatasetUnavailableError` with "PR-9" / "PR-11" messages.
- `judge/prompts.py` — `fingerprint(template)` = sha256 of UTF-8 bytes. `combined_fingerprint(dict)` = order-independent bundle digest (keys sorted, `\x1e`-joined) that will feed the judge cache key in PR-7.
- `judge/longmemeval.py` — five byte-exact templates (`LME_GENERAL_TEMPLATE`, `LME_TEMPORAL_TEMPLATE`, `LME_KNOWLEDGE_UPDATE_TEMPLATE`, `LME_PREFERENCE_TEMPLATE`, `LME_ABSTENTION_TEMPLATE`). `longmemeval_anscheck_prompt(task, q, a, r, *, abstention)` routes task → template and calls `.format(q, a, r)`. `is_abstention_question(qid)` implements the upstream `"_abs" in qid` convention. `parse_yes_no` uses a word-boundary regex (`yes`, not `yesterday`).
- `judge/__init__.py` — re-exports the public surface.

**Byte-exact verification:** each of the five templates was sha256'd and cross-verified against the predecessor `~/code/agent-memory/benchmark/judge.py` at PR-6 commit time. Combined bundle fingerprint: `33013a6ed6390a0d3aaf520ab1c1fda47c345241b34a47a007ae2362d2eb5628`.

**Tests (61 new; 209 total):**
- `test_judge_prompts_stable.py` — parametrized SHA-256 lock per template + combined-bundle lock + placeholder-count + trailing-sentence invariants + `combined_fingerprint` order-independence.
- `test_judge_longmemeval.py` — task-routing dispatch, abstention override, unsupported-task error, `is_abstention_question`, yes/no parser with word-boundary edge cases.
- `test_longmemeval_loader.py` — row-to-case conversion (turn IDs, evidence, integer/None answers, abstention metadata, whitespace stripping, length-mismatch guard), stratified vs. head selection, descriptor-hash stability + drift detection across revision/split/limit/strategy, `load_dataset` dispatcher errors.

**Drive-by fix:** `adapters/base.py` — `# noqa: B027` on the no-op default `close()` (ruff B027 flagged it on the new ruff pin used in this session; the no-op default is intentional because not every adapter needs teardown).

### Current State

- Branch: `feat/longmemeval` (not yet merged). Single commit to follow the handoff commit.
- Tests: `pytest tests/unit -q` → 209 passed.
- Lint: `ruff check`, `ruff format --check` → clean.
- Types: `mypy src` → clean (26 source files).

### What's Next

- **PR-7** — Runner + manifest + scorecard. First end-to-end demo wires `FullContextAdapter` + `OllamaProvider` + `JudgeClient` + LongMemEval loader + cache + manifest. `amb run longmemeval --memory full-context --answer-model ollama:llama3.1:8b --judge-model ollama:llama3.1:70b --split s --limit 5`.
- PR-8 → PR-14 unchanged.

### Open Questions

- Still open: BEAM evidence-turn field name (PR-11); Ollama-digest-in-manifest timing — finalize when PR-7 wires `provider.resolve_spec()`.
- **Revision re-pin cadence.** We pinned `98d7416c…` as the current HF head at PR-6 time. If upstream publishes fixes, re-pin happens as a deliberate migration (descriptor_hash changes → caches invalidate). Document when/if we bump.

### Gotchas

- **`datasets` import is lazy.** `from datasets import load_dataset` happens inside the loader functions, not at module scope, because pandas is blocked by Windows App Control on this host. Tests build HF-shaped dicts by hand and exercise `_row_to_case` + `LongMemEvalDataset.__init__` directly — no round-trip through HF.
- **Evidence granularity is session-level in upstream LongMemEval.** The loader expands every turn inside an `answer_session_ids` session into `evidence_turn_ids` because that's the best the dataset annotates. Turn-level evidence metrics on LongMemEval therefore measure "did we retrieve something from the right *session*", not "the exact right turn".
- **Abstention routing is via question_id, not task.** `"_abs" in question_id` is the upstream convention; `longmemeval_anscheck_prompt` takes the flag as a bool so the caller (runner) can compute it once per question.
- **Re-baselining judge prompts is a migration, not an edit.** Any change to a template requires: bump the golden fingerprint in `test_judge_prompts_stable.py`, bump `protocol_version`, note the migration in `docs/methodology.md`. The fingerprint flows into the judge cache key so prior entries auto-invalidate.

### How to pick up from here

```
cd ~/code/agent-memory-benchmark
source .venv/Scripts/activate
git checkout main          # once PR-6 is merged
git checkout -b feat/runner
# Start PR-7: orchestrator + manifest + scorecard; first amb run demo.
```

---

## Session: 2026-04-20 — PR-5 adapters

### What Was Done

**PR-5 adapters (on `feat/adapters`, branched from `main` after the PR-4 merge):**

- `src/agent_memory_benchmark/adapters/base.py` — `MemoryAdapter` ABC. Sets `memory_system_id` / `memory_version` on subclasses, declares the three abstract async methods (`ingest_session` / `answer_question` / `reset`), and provides default-raising `save_state` / `load_state`. `supports_persistence` is reflection-based (compares the subclass's method objects against the base defaults) so override detection is automatic. `__aexit__` routes through `close()`.
- `adapters/full_context.py` — `FullContextAdapter`. Null-memory baseline: concatenates every ingested turn into the prompt at answer time, emits turn count as `units_retrieved` and a whitespace-token estimate as `tokens_retrieved`, returns an empty `retrieved` tuple. Persistence implemented by JSON-serializing the `{case_id -> [Session]}` dict — the runner can exercise the ingestion cache path even for the baseline.
- `adapters/python_adapter.py` — `PythonAdapter`. Loads `"pkg.module:ClassName"`, instantiates with `--memory-config` kwargs, structurally checks via `isinstance(instance, MemorySystemShape)`. When the check fails, `_describe_missing_shape` emits a targeted "missing attribute/method" diagnostic rather than a bare `isinstance False`. Accepts optional `session_mapper` / `result_mapper` callables (plan option b) so divergent target type shapes bridge without subclassing. Persistence delegates iff the target exposes `save_state` / `load_state`.
- `adapters/factory.py` — `resolve_adapter`. Maps CLI `--memory` to a concrete adapter: `"full-context"` and `"full-context:"` both work; `"python:pkg.module:Class"`; `http://` is reserved with a clear "PR-10" error.

**Tests (47 new; 148 total):** `test_adapters_base.py`, `test_adapters_full_context.py`, `test_adapters_python.py` (incl. module-planting fixture for `from_spec`), `test_adapters_factory.py`.

Also: a small `style(cache)` commit reflowing two already-merged files to match ruff format (one-line calls that fit within 100 cols).

### Current State

- Branch: `feat/adapters` (not yet merged). Head: `style(cache): reflow one-line calls to match ruff format`, built atop `feat(adapters): MemoryAdapter ABC + PythonAdapter + FullContextAdapter`.
- Tests: `pytest` → 148 passed.
- Lint: `ruff check`, `ruff format --check` → clean.
- Types: `mypy src` → clean (20 source files).

### What's Next

- **PR-6** — LongMemEval loader + judge prompts + byte-stable prompt-fingerprint test. Loader ports from `~/code/agent-memory/benchmark/datasets/longmemeval.py` but factors out the HF revision pin. Judge prompts live in `judge/longmemeval.py` + `judge/prompts.py`; fingerprints locked by `test_judge_prompts_stable.py`.
- **PR-7** — Runner + manifest + scorecard. First end-to-end demo will wire `FullContextAdapter` + `OllamaProvider` + `JudgeClient` + LongMemEval loader together.

### Open Questions

- **Type translation strategy locked.** `PythonAdapter` accepts optional `session_mapper` / `result_mapper` kwargs; default is identity (pass-through). Resolves plan open-question #7.
- Still open: BEAM evidence-turn field name (PR-11); Ollama-digest-in-manifest timing decision (plan open-question #3) will be finalized when PR-7 wires `provider.resolve_spec()` through the runner.

### Gotchas

- **`MemorySystemShape` structural check skips async-ness.** `Protocol` `isinstance` validates attribute *presence*, not callability signature. `PythonAdapter._describe_missing_shape` checks `callable(...)` as a best-effort extra, but adapters still need to assume targets were written to the documented contract; a sync `def ingest_session` would pass the structural check and fail at `await`. Documented in `docs/compat.md`.
- **`Protocol.isinstance` vs. `runtime_checkable`.** `MemorySystemShape` is already `@runtime_checkable`; don't remove that decorator or the adapter loses its friendly-error path.
- **FullContextAdapter's `load_state` only handles the format `save_state` writes.** External state files must include `memory_system_id` / `memory_version` / `sessions` keys with the documented session shape, or `KeyError` will surface — that's intentional (fail-loud on schema drift).

### How to pick up from here

```
cd ~/code/agent-memory-benchmark
source .venv/Scripts/activate
git checkout main          # once PR-5 is merged
git checkout -b feat/longmemeval
# Start PR-6: LongMemEval loader + judge prompts + prompt-fingerprint lock.
```

---

## Session: 2026-04-20 — PR-4 cache layer

### What Was Done

**PR-4 cache layer (on `feat/cache-layer`, branched from `main` after the PR-3 merge):**

- `src/agent_memory_benchmark/cache/keys.py` — byte-stable key derivation ported from `~/code/agent-memory/benchmark/cache.py`:
  - `SEPARATOR = b"\x1e"` (ASCII RS) between UTF-8 parts, with a trailing separator after the final part. Part ordering matches the predecessor exactly.
  - `answer_key` takes optional `replicate_idx`; `idx=0` omits the trailing part so single-run digests remain predecessor-compatible. `idx>0` appends `str(idx)` — the one deliberate divergence, kept at the end of the parts tuple.
  - Numeric fields frozen: judge temperature as `f"{t:.6f}"`, runs as `str(int)`. Question-type/id `"{t or ''}|{id or ''}"` framing preserved.
- `cache/index.py` — `cache_index.json` CRUD: `load_index` (corruption-safe), `index_touch`, `CacheIndexWriter` batch writer, `clear_all` / `clear_kind`, and `gc_older_than` for `amb cache gc --before 30d`. GC normalizes the `%Y-%m-%dT%H:%M:%SZ` UTC timestamps we write by subtracting `time.timezone`.
- `cache/m3_guard.py` — publish-time safety: walks `<root>/answers/*.json`, flags any entry whose stored `memory_version` differs from the adapter's current value, or whose payload is non-dict / unparseable. Entries belonging to other memory systems are skipped.
- `.gitignore` — anchored `cache/`, `results/`, `data/` with leading `/` so they no longer shadow same-named Python modules under `src/`. The unanchored pattern was silently blocking `git add src/agent_memory_benchmark/cache/`. Documented in `.agent/lessons.md`.

**Tests (46 new; 101 total):**
- `test_cache_keys.py` — hex-digest goldens for `hash_parts`/`ingestion_key`/`answer_key` (both replicate-idx branches) / `judge_key` (with and without type/id); trailing-separator framing check; UTF-8 multibyte stability; temperature-format equivalences (`0`, `0.0`, `0.000000`); path-layout + slash sanitization.
- `test_cache_index.py` — corruption-safe load, touch/overwrite, batch writer idempotency, clear helpers, `gc_older_than` edge cases (missing-on-disk, unparseable timestamp, absolute-path entries, negative-age rejection).
- `test_cache_m3_guard.py` — empty/all-matching/mismatch/other-system/unreadable/non-dict/missing-version/sorted-output paths.

### Current State

- Branch: `feat/cache-layer` (not yet merged). Head commit: `feat(cache): byte-stable keys, index, and M3 publish guard`.
- Tests: `pytest` → 101 passed.
- Lint: `ruff check`, `ruff format --check` → clean.
- Types: `mypy src` → clean (15 source files).

### What's Next

- **PR-5** — Adapters: `adapters/base.py` `MemoryAdapter` ABC, `adapters/python_adapter.py` with structural-type check against `MemorySystemShape`, `adapters/full_context.py` null baseline. The cache layer is already wired to accept `replicate_idx` so the runner can drive noise-aware replicates once PR-5 lands the adapter plumbing.
- PR-6 → PR-7 → PR-8 unchanged from prior plan.

### Open Questions

- **Replicate semantics locked.** `answer_key(..., replicate_idx=0)` is byte-exact with the predecessor; `replicate_idx>0` appends `str(idx)`. Resolves plan open-question #4.
- Still open: BEAM evidence-turn field name (PR-11); PythonAdapter type translation strategy (PR-5).

### Gotchas

- **`.gitignore` anchor trap.** Unanchored runtime-output ignores (`cache/`) shadow in-package modules with the same name. Always anchor top-level runtime dirs with a leading slash. Captured as the first entry in `.agent/lessons.md`.
- **GC timestamp arithmetic on Windows.** `time.mktime(time.strptime(...))` interprets the struct as **local time**; our stored timestamps are UTC-`Z`. The fix is `epoch -= time.timezone` after `mktime`. Verified in `test_removes_old_and_keeps_fresh`.
- **Cache-key byte-exactness locks are real.** The hex-digest goldens in `test_cache_keys.py` will fail noisily on any accidental algorithm drift. Pair that with a migration note + bumped digests if the change is intentional.

### How to pick up from here

```
cd ~/code/agent-memory-benchmark
source .venv/Scripts/activate
git checkout main       # once PR-4 is merged
git checkout -b feat/adapters
# Start PR-5: MemoryAdapter ABC + PythonAdapter + FullContextAdapter.
```

---

## Session: 2026-04-20 — PR-3 LLM providers

### What Was Done

**PR-3 LLM providers (on `feat/llm-providers`, branched from `main`):**
- `src/agent_memory_benchmark/llm/__init__.py` — `LLMProvider` Protocol (runtime_checkable), `ChatResult` dataclass, `parse_spec` (first-colon split so `ollama:llama3.1:8b` parses correctly), `build_provider` factory, error hierarchy (`ProviderError` / `ProviderUnavailableError` / `InvalidSpecError`).
- `llm/ollama.py` — `OllamaProvider` over httpx on `/api/chat`, native JSON mode via `format:"json"`, `num_ctx`/`num_predict` options, `resolve_spec()` pins `ollama:<model>@sha256:<digest>` via `/api/tags` (falls back to unpinned when model not yet pulled). `OllamaError` carries `status_code` for clean retry classification.
- `llm/openai.py` — `OpenAIProvider` backed by the `openai` SDK, lazy import so the `[openai]` extra stays optional; OpenAI dated model IDs are already immutable so `resolve_spec` echoes the spec.
- `llm/judge_client.py` — `JudgeClient` facade: retries on 429/5xx/transport errors with exponential backoff + ±25% jitter (capped at 30s), concurrent `complete_runs()` for LOCOMO-style majority voting.

**Tests (38 new):**
- `tests/unit/test_llm_spec.py` — `parse_spec` edge cases (colon-containing models, digest pins, malformed input), `build_provider` error paths including a `ModuleNotFoundError` injection that only intercepts the **absolute** `openai` import (level==0) so the relative `.openai` import still resolves.
- `tests/unit/test_llm_ollama.py` — `/api/chat` payload shape, JSON mode, HTTP error propagation, `/api/tags` digest resolution with both `sha256:` and bare-digest forms, fallback when model not listed, owned vs. injected client lifecycle. Uses `respx`.
- `tests/unit/test_llm_openai.py` — `importorskip`-guarded; exercises the SDK's actual request/response plumbing through `respx`, system-omission, JSON mode, missing-key error.
- `tests/unit/test_llm_judge_client.py` — retry classifier including transient (429, 503, httpx timeout) vs. non-transient (400); `asyncio.sleep` monkeypatched to zero for instant retry tests; multi-run `gather` fanout.

### Current State

- Branch: `feat/llm-providers` (not yet merged to main). Head: `feat(llm): Ollama/OpenAI providers + judge facade`.
- Tests: `pytest` → 55 passed (17 existing + 38 new).
- Lint: `ruff check`, `ruff format --check` → clean.
- Types: `mypy src` → clean.
- CLI: `amb --version` still works; no CLI surface changed in this PR.

### What's Next

- **PR-4** — Cache layer (`cache/keys.py`, `cache/index.py`, `cache/m3_guard.py`). Port SHA-256 inputs byte-exact from `~/code/agent-memory/benchmark/cache.py` and lock each with a regression test. Cache keys will use `provider.resolve_spec()` so Ollama digest pinning flows through.
- **PR-5** — Adapters (`adapters/base.py` MemoryAdapter ABC, `adapters/python_adapter.py` structural-type check, `adapters/full_context.py` null baseline).
- PR-6 → PR-7 → PR-8 as previously laid out.

### Open Questions

- **Ollama digest resolution timing** — decided in this PR to expose `resolve_spec()` as an async method on each provider and let the runner call it once at run start. The cache layer (PR-4) will hash the resolved spec.
- Still open (unchanged from prior handoff): BEAM evidence-turn field name (PR-11), PythonAdapter type translation (PR-5).

### Gotchas

- **Relative-import trap**: `from .openai import OpenAIProvider` is a level-1 relative import. When monkeypatching `builtins.__import__` to simulate a missing `openai` SDK, only intercept when `level == 0` — otherwise the provider's own submodule import is killed too. Documented in `test_openai_without_sdk_raises_provider_unavailable`.
- **Ollama digest via `/api/tags`, not `/api/show`**: `/api/show`'s digest field is not uniformly present across Ollama versions; `/api/tags` lists all local models with `digest` reliably. If the model hasn't been pulled yet, `resolve_spec` returns the unpinned spec — the M3 guard (PR-4) is the right place to hard-fail on that for `--publishable` runs.
- **respx intercepts the OpenAI SDK's httpx calls automatically** once the SDK is using its default httpx transport. Set a distinct `base_url` in tests (`https://api.openai.test/v1`) to avoid masking real endpoints in reviewers' local runs.

### How to pick up from here

```
cd ~/code/agent-memory-benchmark
source .venv/Scripts/activate
git checkout main      # if PR-3 has been merged
# or stay on feat/llm-providers while PR-3 waits for review
git checkout -b feat/cache-layer
# Port byte-exact hash inputs from ~/code/agent-memory/benchmark/cache.py
```

---

## Session: 2026-04-20 — repo init + PR-1 + PR-2

### What Was Done

**Repo init (on `main`):**
- `git init -b main` at `~/code/agent-memory-benchmark`
- Remote set to `https://github.com/rtuosto/agent-memory-benchmark.git` (not yet pushed)
- Initial commit: LICENSE (MIT), .gitignore, README stub

**Agent bootstrap (on `feat/scaffold`):**
- Ran `~/code/agent-bootstrap/setup.sh` — installed CLAUDE.md, .cursorrules, .agent/*, docs/ARCHITECTURE.md
- Customized CLAUDE.md "Project-Specific Context" with stack, invariants, KPI catalog, user context, design posture
- Replaced docs/ARCHITECTURE.md template with project-specific technical map
- Corrected .gitignore — only `current-plan.md` is ignored; `session-handoff.md` and `lessons.md` are committed

**PR-1 scaffold (on `feat/scaffold`):**
- `pyproject.toml` (hatchling, strict mypy, ruff, pytest-asyncio, extras [openai]/[tiktoken]/[dev])
- `src/agent_memory_benchmark/{__init__,__main__,version,py.typed}`
- `src/agent_memory_benchmark/cli/{__init__,main}.py` — argparse root; subcommands stubbed for PR-8
- `tests/unit/test_version.py` — smoke tests
- `.github/workflows/ci.yml` — ruff + ruff format --check + mypy + pytest on ubuntu/windows × py3.11/3.12

**PR-2 types + compat (on `feat/types-compat`, stacked on feat/scaffold):**
- `src/agent_memory_benchmark/types.py` — Turn, Session (frozen), QAItem (with evidence_turn_ids), BenchmarkCase, RetrievedUnit (with source_turn_ids), AnswerResult; DatasetName Literal
- `src/agent_memory_benchmark/compat.py` — MemorySystemShape (runtime_checkable Protocol) + PersistableMemorySystemShape
- `src/agent_memory_benchmark/__init__.py` re-exports public types
- `tests/unit/test_types.py` and `tests/unit/test_compat_shape.py`
- `docs/compat.md` — adapter-compatibility spec for memory-system authors

### Current State

- Branches: `main` has PR-1 + PR-2 merged in (user-approved merge). `feat/scaffold` and `feat/types-compat` are retained locally as merge-history audit trails but are no longer needed for active work.
- `main` head: contains the full bootstrap + scaffold + types/compat layers (5 commits from repo init).
- Remote: `https://github.com/rtuosto/agent-memory-benchmark.git` set as `origin` but nothing pushed yet.
- Tests: `pytest` → 17 passed on `main`.
- Lint: `ruff check`, `ruff format --check` → clean.
- Types: `mypy src` → clean.
- CLI: `amb --version` → `amb 0.1.0`.
- Venv: `.venv/` at repo root; `pip install -e ".[dev,openai,tiktoken]"` succeeds.

### What's Next

- **PR-3** — LLM providers (`llm/ollama.py` httpx client, `llm/openai.py` behind [openai], `llm/judge_client.py` facade). Use `respx` fixtures for tests.
- **PR-4** — Cache layer (`cache/keys.py`, `cache/index.py`, `cache/m3_guard.py`). Port SHA-256 inputs byte-exact from `~/code/agent-memory/benchmark/cache.py`.
- **PR-5** — Adapters (`adapters/base.py` MemoryAdapter ABC, `adapters/python_adapter.py` structural-type check, `adapters/full_context.py` null baseline).
- **PR-6** — LongMemEval loader + judge prompts + byte-stable prompt-fingerprint test.
- **PR-7** — Runner + manifest + scorecard. First runnable demo: `amb run longmemeval --memory full-context --answer-model ollama:llama3.1:8b --judge-model ollama:llama3.1:70b --split s --limit 5`.

### Open Questions

- **BEAM evidence-turn field name** on HF `Mohammadta/BEAM` — confirm at PR-11 against actual HF schema.
- **PythonAdapter type translation** when a memory system uses its own Session / AnswerResult field names — leaning toward `--memory-config mapper=pkg.mod:fn` but will finalize at PR-5.
- **Ollama digest in cache key** — resolve at runtime and encode as `ollama:llama3.1:8b@sha256:...`. Need to decide in PR-3 whether to always resolve or opt in.

### Gotchas

- `.gitignore` initially had `.agent/session-handoff.md` listed — that was wrong. Fixed in the bootstrap commit on `feat/scaffold`.
- Windows paths: all shell uses forward slashes + bash. `results/latest` symlink will need Windows-junction fallback logic in PR-7.
- `gh` CLI is not installed — branches push via `git push -u origin <branch>` and PRs are opened manually through the GitHub web UI.
- Stacked-branch layout means feat/types-compat must be rebased onto main after feat/scaffold merges.
- Venv at `.venv/` is gitignored; any new shell needs `source .venv/Scripts/activate` before `pytest`.

### How to pick up from here

```
cd ~/code/agent-memory-benchmark
source .venv/Scripts/activate
git checkout main
git pull --ff-only origin main   # if/when pushed
# Start PR-3 work:
git checkout -b feat/llm-providers
```
