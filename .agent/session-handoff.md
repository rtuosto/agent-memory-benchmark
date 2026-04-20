# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

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

- Branches (local only; not pushed):
  - `main` — 1 commit (bootstrap stub)
  - `feat/scaffold` — 2 commits on top of main (bootstrap + scaffold)
  - `feat/types-compat` — 1 commit on top of feat/scaffold (types + compat)
- Tests: `pytest` → 17 passed
- Lint: `ruff check`, `ruff format --check` → clean
- Types: `mypy src` → clean
- CLI: `amb --version` → `amb 0.1.0`
- Venv: `.venv/` at repo root; `pip install -e ".[dev,openai,tiktoken]"` succeeds

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
git checkout feat/types-compat
# Start PR-3 work on a new branch stacked on feat/types-compat:
git checkout -b feat/llm-providers
```
