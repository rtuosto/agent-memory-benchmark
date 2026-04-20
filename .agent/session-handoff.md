# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

---

## Session: 2026-04-20 — PR-1 scaffold

### What Was Done
- Repo initialized (`main` branch) with `LICENSE` (MIT), `.gitignore`, stub `README.md`
- Ran agent bootstrap (`~/code/agent-bootstrap/setup.sh`) — installed `CLAUDE.md`, `.cursorrules`, `.agent/*`, `docs/ARCHITECTURE.md`
- Customized `CLAUDE.md` "Project-Specific Context" with stack, invariants, KPI catalog, user context
- Replaced `docs/ARCHITECTURE.md` template with project-specific technical map
- Fixed `.gitignore` — `session-handoff.md` and `lessons.md` are committed; only `current-plan.md` is ignored
- Built PR-1 scaffold on `feat/scaffold`:
  - `pyproject.toml` — hatchling build, strict mypy, ruff, pytest-asyncio, extras `[openai]` / `[tiktoken]` / `[dev]`
  - `src/agent_memory_benchmark/{__init__,__main__,version,py.typed}`
  - `src/agent_memory_benchmark/cli/{__init__,main}.py` — argparse root with `amb --version` working; subcommands stubbed for PR-8
  - `tests/unit/test_version.py` — smoke test for package import and CLI `--version`
  - `.github/workflows/ci.yml` — ruff, mypy, pytest on Ubuntu + Windows, py 3.11 + 3.12

### Current State
- Branch: `feat/scaffold`
- Tests: not yet run locally (need `pip install -e ".[dev]"`)
- Build: `pyproject.toml` is valid hatchling config; `amb --version` should produce `amb 0.1.0`

### What's Next
- PR-2: `src/agent_memory_benchmark/types.py` + `compat.py` (MemorySystemShape Protocol) + unit tests
- PR-3: LLM providers (`llm/ollama.py`, `llm/openai.py`, `llm/judge_client.py`) with `respx` fixtures
- PR-4: Cache layer (`cache/keys.py`, `cache/index.py`, `cache/m3_guard.py`) — port key inputs byte-exact from `~/code/agent-memory/benchmark/cache.py`
- PR-5: Adapters (`adapters/base.py`, `adapters/python_adapter.py`, `adapters/full_context.py`)
- PR-6: LongMemEval loader + judge prompts with byte-stable fingerprint tests
- PR-7: Runner + scorecard → first runnable `amb run longmemeval --memory full-context ...` demo

### Open Questions
- BEAM evidence-turn field name on HF `Mohammadta/BEAM` — confirm at PR-11.
- Exact type translation for `PythonAdapter` when engram's `Session`/`AnswerResult` field names differ slightly from the benchmark's — defer to PR-5 (likely: optional mapper function in `--memory-config`).

### Gotchas
- `.gitignore` initially had `.agent/session-handoff.md` listed — that was a mistake. Per bootstrap design, only `current-plan.md` is ignored. Fixed on this branch.
- Windows paths: all file operations use forward slashes in bash; `results/latest` symlink will need Windows-junction fallback logic when PR-7 lands.
- `gh` CLI is not installed on this machine — PRs are opened manually. Branches push normally via `git push -u origin <branch>`.
