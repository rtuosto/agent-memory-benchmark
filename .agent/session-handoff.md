# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

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
