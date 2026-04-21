# Agent Lessons Learned

> Read this file at the START of every session. These are mistakes that have already been made.
> Don't repeat them. Add new entries when things go wrong.

---

## 2026-04-20 — .gitignore `cache/` swallowed the source package

**What happened:** `git add src/agent_memory_benchmark/cache/` failed silently with "paths are ignored by one of your .gitignore files". The `.gitignore` had a bare `cache/` entry intended for the user-facing runtime cache at repo root, but gitignore patterns without a leading slash match at any depth — so it matched the in-package `src/agent_memory_benchmark/cache/` module too.

**Root cause:** Ambiguous gitignore. A pattern like `cache/` with no anchor matches `cache/` anywhere in the tree. Same class of bug applied to `results/` and `data/`.

**What I should have done:** Anchor runtime-output ignores to the repo root from the start, e.g. `/cache/` not `cache/`.

**Rule:** In this repo, runtime-output directory ignores (`cache/`, `results/`, `data/`, etc.) MUST be anchored with a leading slash (`/cache/`) so they do not shadow same-named Python modules inside `src/`. When adding a new ignored directory, ask: "does this name appear — or might appear — anywhere under `src/`?" If yes, anchor it.

<!--
Entry format:

## [DATE] — [Short Title]

**What happened:** [Brief description]
**Root cause:** [Why — be specific]  
**What I should have done:** [Correct approach]
**Rule:** [Concrete, actionable rule to prevent recurrence]
-->
