# Compatibility with `PythonAdapter`

This document describes the shape a memory system's Python class needs to
match so that `agent-memory-benchmark`'s `PythonAdapter` can drive it in
process. **You do not import anything from this repo to be compatible.** The
shape below is documentation; `agent_memory_benchmark.compat.MemorySystemShape`
is a published `typing.Protocol` the adapter uses internally for structural
type checks.

If you prefer a different transport:

- **HTTP:** implement the REST endpoints documented in `docs/http-api.md` /
  `openapi.yaml` — no Python coupling at all.
- **Subprocess / CLI:** not yet supported; file an issue if you need it.

## The contract

```python
# Example of a compatible class — nothing below is imported from this repo.

class MyMemory:
    memory_system_id = "mymemory"  # short, stable slug
    memory_version = "0.1.0"       # bump on semantics / state-format change

    async def ingest_session(self, session, case_id: str) -> None:
        """Append one session to the memory for this case.

        Called in chronological order per case_id. Deterministic: same inputs
        must produce the same state.
        """
        ...

    async def answer_question(self, question: str, case_id: str) -> "AnswerResult":
        """Retrieve + generate an answer for this question.

        Return an object with at least the following attributes:

        - answer: str
        - retrieval_time_ms: float
        - generation_time_ms: float
        - units_retrieved: int = 0
        - tokens_retrieved: int = 0
        - retrieved: tuple of RetrievedUnit = ()

        The runner independently measures total wall time; your self-reports
        are stored alongside but not substituted.
        """
        ...

    async def reset(self) -> None:
        """Clear all ingested state before the next case."""
        ...
```

Optional — implementing these enables the ingestion cache:

```python
    async def save_state(self, path: Path) -> None: ...
    async def load_state(self, path: Path) -> None: ...
```

## Types the adapter passes in

`session` is an immutable dataclass with:

- `session_index: int` — 1-based, chronological within the case
- `turns: tuple[Turn, ...]` — each Turn has `turn_id`, `speaker`, `text`,
  optional `timestamp`, optional `image_caption`
- `session_time: str | None`
- `session_id: str | None`

## Types the adapter expects back

`AnswerResult` expects the attributes listed above. If your system uses its
own classes with different names, supply mapper callables (see next section)
that translate at the adapter boundary.

## Mapper functions (for divergent type shapes)

`PythonAdapter` accepts two optional callables:

- `session_mapper: Callable[[Session], Any]` — converts the benchmark's
  `Session` into whatever type your `ingest_session` expects.
- `result_mapper: Callable[[Any], AnswerResult]` — converts whatever your
  `answer_question` returns into the benchmark's `AnswerResult`.

These are resolved from the CLI as module-qualified function names using the
same grammar as `--memory python:pkg.mod:Class`:

```bash
amb run longmemeval \
    --memory python:yourpackage.mymem:MyMemory \
    --session-mapper yourpackage.shim:to_your_session \
    --result-mapper yourpackage.shim:from_your_result \
    --memory-config embedding_model=... \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b
```

Both flags default to identity (pass-through). If your types already match
the benchmark's shape field-for-field, you don't need either.

## Reference: the engram shim

Engram ships a multi-layer memory system at `memory.system.MultiLayerMemory`
whose `Session` / `Turn` / `AnswerResult` field names diverge from the
benchmark's (`DialogueTurn.dia_id` vs. `Turn.turn_id`,
`Session.date_time` vs. `Session.session_time`, etc.). The shim at
`agent_memory_benchmark.compat.engram_shim` provides matched mappers:

```bash
amb run longmemeval \
    --memory python:memory.system:MultiLayerMemory \
    --memory-config embedding_model=sentence-transformers/all-MiniLM-L6-v2 \
    --session-mapper agent_memory_benchmark.compat.engram_shim:to_engram_session \
    --result-mapper agent_memory_benchmark.compat.engram_shim:from_engram_answer \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b \
    --split s --limit 5
```

A paired engram-side patch adds `memory_system_id` and `memory_version` as
`ClassVar[str]` on `MultiLayerMemory` (a module-level constant does not
satisfy the `isinstance(..., MemorySystemShape)` check). Without that patch
the adapter raises `PythonAdapterError` at construction time with the
specific missing attributes listed.

Evidence-keyed KPIs will report `null` until engram populates
`RetrievedUnit.source_turn_ids` on returned units. Quality, latency,
footprint, and throughput KPIs all work in the meantime.

## Evidence-keyed retrieval KPIs

If you populate `retrieved` with `RetrievedUnit` instances that carry
`source_turn_ids`, the scorer computes turn-level, unit-level, and token-level
evidence KPIs. When `source_turn_ids` is absent, the scorer falls back to
substring matching against known turn text; severe mismatch causes turn- and
unit-level metrics to be reported as `null` for that question. Token metrics
always work if you populate `text`.

## Pointing the benchmark at your class

```bash
amb run longmemeval \
    --memory python:yourpackage.mymem:MyMemory \
    --memory-config embedding_model=... \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b
```

`--memory-config key=value` entries are passed as keyword arguments to your
class's `__init__`. Values are parsed as JSON when the string is valid JSON,
otherwise kept as strings (so `timeout=30` is an int, `model=llama3` stays a
string). Use `--session-mapper` / `--result-mapper` for type translation at
the adapter boundary; see the section above.

## What "structural typing" buys you

The benchmark's `MemorySystemShape` Protocol is `runtime_checkable`. Before
the first call, `PythonAdapter` runs `isinstance(instance, MemorySystemShape)`
and produces a friendly error if the shape doesn't match. This check doesn't
verify argument *types* or that methods are async — targeted attribute
inspection handles the rest.

## What this contract does NOT require

- You do not inherit from any benchmark class.
- You do not import any benchmark module.
- You do not need to speak the benchmark's `AnswerResult` type directly if
  you supply a mapper.
- You do not need to implement save/load state — but you lose the ingestion
  cache if you don't.
