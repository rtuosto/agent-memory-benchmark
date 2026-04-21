# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

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
