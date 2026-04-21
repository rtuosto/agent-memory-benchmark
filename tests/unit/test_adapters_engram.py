"""Tests for :class:`EngramAdapter` — the bridge from the benchmark runner
to ``engram.EngramGraphMemorySystem``.

These tests stub out both engram and the LLM provider so they don't need
the real engram package installed. They verify the boundary translation:

- one :class:`engram.Memory` per :class:`Turn` with expected metadata
- ``answer_question`` calls ``engram.recall`` then the LLM, assembles
  :class:`AnswerResult` with the right timing / size fields
- ``reset`` / ``save_state`` / ``load_state`` delegate
- the factory resolves ``--memory engram``
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_memory_benchmark.adapters.engram_adapter import (
    EngramAdapter,
    _format_recall_context,
    _normalize_timestamp,
    _passages_to_units,
)
from agent_memory_benchmark.adapters.factory import AdapterSpecError, resolve_adapter
from agent_memory_benchmark.llm import ChatResult
from agent_memory_benchmark.types import Session, Turn

# -------------------------------- fakes -----------------------------------


@dataclass(frozen=True)
class _FakeMemory:
    content: str
    timestamp: str | None = None
    speaker: str | None = None
    source: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _FakePassage:
    text: str
    granularity: str = "turn"
    score: float = 1.0
    node_id: str = "n0"
    source_memory_id: str | None = None
    source_memory_index: int | None = None
    timestamp: str | None = None
    speaker: str | None = None
    supporting_edges: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FakeFact:
    kind: str
    subject: str
    value: str
    predicate: str | None = None
    object: str | None = None
    timestamp: str | None = None
    supporting_memory_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FakeRecallResult:
    passages: tuple[_FakePassage, ...] = ()
    facts: tuple[_FakeFact, ...] = ()
    intent: str | None = None
    intent_confidence: float = 0.0
    timing_ms: tuple[tuple[str, float], ...] = ()
    recall_fingerprint: str | None = None


class _FakeEngram:
    memory_system_id = "engram_graph"
    memory_version = "0.4.0"

    def __init__(self, recall_result: _FakeRecallResult | None = None) -> None:
        self.ingested: list[_FakeMemory] = []
        self.ingest_many_calls = 0
        self.reset_calls = 0
        self.save_calls: list[Path] = []
        self.load_calls: list[Path] = []
        self._recall_result = recall_result or _FakeRecallResult()

    async def ingest(self, memory: _FakeMemory) -> None:
        self.ingested.append(memory)

    async def ingest_many(self, memories: object) -> None:
        # Mirrors engram's MemorySystem.ingest_many default (loop of ingest)
        # while letting tests assert the adapter used the batched entry point
        # by inspecting ``ingest_many_calls``.
        self.ingest_many_calls += 1
        for memory in memories:  # type: ignore[attr-defined]
            self.ingested.append(memory)

    async def recall(
        self,
        query: str,
        *,
        now: str | None = None,
        timezone: str | None = None,
        max_passages: int | None = None,
        intent_hint: str | None = None,
    ) -> _FakeRecallResult:
        self.last_query = query
        self.last_max_passages = max_passages
        return self._recall_result

    async def reset(self) -> None:
        self.reset_calls += 1

    async def save_state(self, path: Path) -> None:
        self.save_calls.append(path)

    async def load_state(self, path: Path) -> None:
        self.load_calls.append(path)


class _FakeProvider:
    spec = "fake:answerer"
    model = "answerer"

    def __init__(self, reply: str = "reply") -> None:
        self._reply = reply
        self.last_user: str | None = None
        self.last_system: str | None = None
        self.close_calls = 0

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        self.last_user = user
        self.last_system = system
        return ChatResult(text=self._reply, model=self.model)

    async def resolve_spec(self) -> str:
        return self.spec

    async def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def fake_engram_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plant a fake ``engram`` module in ``sys.modules`` so ``from engram import Memory``
    inside ``EngramAdapter.ingest_session`` picks up our fake type.

    Leaves ``EngramGraphMemorySystem`` off the module so any caller that
    forgets to inject ``target=...`` fails loudly rather than instantiating
    the real thing in a unit test."""

    module = types.ModuleType("engram")
    module.Memory = _FakeMemory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "engram", module)


def _session(session_index: int = 1) -> Session:
    return Session(
        session_index=session_index,
        turns=(
            Turn(turn_id="t1", speaker="alice", text="hello", timestamp="2024-01-01T10:00"),
            Turn(
                turn_id="t2",
                speaker="bob",
                text="hi back",
                timestamp=None,
                image_caption="a beach",
            ),
        ),
        session_time="2024-01-01",
        session_id="s1",
    )


# ---------------------------- ingestion -----------------------------------


def test_ingest_session_emits_one_memory_per_turn(fake_engram_module: None) -> None:
    engram = _FakeEngram()
    adapter = EngramAdapter(_FakeProvider(), target=engram)

    asyncio.run(adapter.ingest_session(_session(), case_id="case_a"))

    assert len(engram.ingested) == 2
    first, second = engram.ingested
    assert first.content == "hello"
    assert first.speaker == "alice"
    assert first.timestamp == "2024-01-01T10:00:00"
    assert first.source == "conversation_turn"
    assert second.content == "hi back"
    # turn has no timestamp of its own → falls back to session_time.
    assert second.timestamp == "2024-01-01T00:00:00"


def test_ingest_session_uses_batched_ingest_many(fake_engram_module: None) -> None:
    """One ``ingest_many`` call per session — that's what unlocks engram's
    pooled transformer forwards. Regressing to a per-turn ``ingest`` loop
    would silently drop the ~2.6× ingestion speedup."""

    engram = _FakeEngram()
    adapter = EngramAdapter(_FakeProvider(), target=engram)

    asyncio.run(adapter.ingest_session(_session(), case_id="case_a"))

    assert engram.ingest_many_calls == 1
    assert len(engram.ingested) == 2


def test_ingest_session_metadata_sorted_and_includes_case_id(fake_engram_module: None) -> None:
    engram = _FakeEngram()
    adapter = EngramAdapter(_FakeProvider(), target=engram)

    asyncio.run(adapter.ingest_session(_session(), case_id="case_a"))

    first_meta = dict(engram.ingested[0].metadata)
    assert first_meta["case_id"] == "case_a"
    assert first_meta["turn_id"] == "t1"
    assert first_meta["session_index"] == "1"
    assert first_meta["session_id"] == "s1"
    # Sorted invariant — metadata is hashable and deterministic.
    keys = [k for k, _ in engram.ingested[0].metadata]
    assert keys == sorted(keys)


def test_ingest_session_normalizes_longmemeval_timestamp(fake_engram_module: None) -> None:
    """LongMemEval ships timestamps like ``2023/05/20 (Sat) 02:21`` — engram
    rejects those with ``fromisoformat``. The adapter must coerce to ISO."""

    engram = _FakeEngram()
    adapter = EngramAdapter(_FakeProvider(), target=engram)
    session = Session(
        session_index=1,
        turns=(Turn(turn_id="t1", speaker="alice", text="hi", timestamp=None),),
        session_time="2023/05/20 (Sat) 02:21",
    )
    asyncio.run(adapter.ingest_session(session, case_id="c"))
    ts = engram.ingested[0].timestamp
    assert ts is not None
    assert ts.startswith("2023-05-20T02:21")


def test_ingest_session_drops_unparseable_timestamp(fake_engram_module: None) -> None:
    engram = _FakeEngram()
    adapter = EngramAdapter(_FakeProvider(), target=engram)
    session = Session(
        session_index=1,
        turns=(Turn(turn_id="t1", speaker="a", text="hi", timestamp="not-a-date-at-all"),),
        session_time=None,
    )
    asyncio.run(adapter.ingest_session(session, case_id="c"))
    assert engram.ingested[0].timestamp is None


@pytest.mark.parametrize(
    "raw,expected_prefix",
    [
        ("2024-01-01T10:00", "2024-01-01T10:00"),
        ("2023/05/20 (Sat) 02:21", "2023-05-20T02:21"),
        ("2024-01-01", "2024-01-01T00:00:00"),
    ],
)
def test_normalize_timestamp_parses_common_formats(raw: str, expected_prefix: str) -> None:
    result = _normalize_timestamp(raw)
    assert result is not None
    assert result.startswith(expected_prefix)


@pytest.mark.parametrize("raw", [None, "", "   ", "totally unparseable gibberish"])
def test_normalize_timestamp_drops_empty_or_unparseable(raw: str | None) -> None:
    assert _normalize_timestamp(raw) is None


def test_ingest_session_propagates_image_caption(fake_engram_module: None) -> None:
    engram = _FakeEngram()
    adapter = EngramAdapter(_FakeProvider(), target=engram)

    asyncio.run(adapter.ingest_session(_session(), case_id="case_a"))
    second_meta = dict(engram.ingested[1].metadata)
    assert second_meta["image_caption"] == "a beach"


# ---------------------------- answer --------------------------------------


def test_answer_question_recalls_then_generates(fake_engram_module: None) -> None:
    recall = _FakeRecallResult(
        passages=(
            _FakePassage(text="alice said hello", node_id="p1", score=0.9, speaker="alice"),
            _FakePassage(text="bob replied hi back", node_id="p2", score=0.7, speaker="bob"),
        ),
        facts=(_FakeFact(kind="current_preference", subject="alice", value="likes tea"),),
    )
    engram = _FakeEngram(recall_result=recall)
    provider = _FakeProvider(reply="the answer")
    adapter = EngramAdapter(provider, target=engram, max_passages=5)

    result = asyncio.run(adapter.answer_question("what did alice say?", case_id="case_a"))

    assert engram.last_query == "what did alice say?"
    assert engram.last_max_passages == 5
    assert result.answer == "the answer"
    assert result.units_retrieved == 2
    # whitespace token estimate over passage text: 3 + 4 = 7
    assert result.tokens_retrieved == 7
    assert len(result.retrieved) == 2
    assert result.retrieved[0].unit_id == "p1"
    assert result.retrieved[0].score == 0.9
    assert result.retrieval_time_ms >= 0.0
    assert result.generation_time_ms >= 0.0

    # Prompt composition — context must carry facts + passages to the LLM.
    assert provider.last_user is not None
    assert "what did alice say?" in provider.last_user
    assert "alice said hello" in provider.last_user
    assert "likes tea" in provider.last_user


def test_answer_question_on_empty_recall_shows_placeholder(fake_engram_module: None) -> None:
    engram = _FakeEngram(recall_result=_FakeRecallResult())
    provider = _FakeProvider()
    adapter = EngramAdapter(provider, target=engram)

    result = asyncio.run(adapter.answer_question("q", case_id="c"))
    assert result.units_retrieved == 0
    assert result.tokens_retrieved == 0
    assert result.retrieved == ()
    assert provider.last_user is not None
    assert "no memories retrieved" in provider.last_user


# ---------------------------- lifecycle -----------------------------------


def test_reset_delegates() -> None:
    engram = _FakeEngram()
    adapter = EngramAdapter(_FakeProvider(), target=engram)
    asyncio.run(adapter.reset())
    assert engram.reset_calls == 1


def test_save_and_load_state_delegate(tmp_path: Path) -> None:
    engram = _FakeEngram()
    adapter = EngramAdapter(_FakeProvider(), target=engram)
    asyncio.run(adapter.save_state(tmp_path / "state"))
    asyncio.run(adapter.load_state(tmp_path / "state"))
    assert engram.save_calls == [tmp_path / "state"]
    assert engram.load_calls == [tmp_path / "state"]


def test_save_state_writes_state_json_sentinel(tmp_path: Path) -> None:
    """Runner's ingestion cache keys on ``<dir>/state.json``. Without
    this sentinel every repeat run re-ingests even though engram's
    primary/embeddings/manifest files are on disk."""

    import json as _json

    engram = _FakeEngram()
    adapter = EngramAdapter(_FakeProvider(), target=engram)
    asyncio.run(adapter.save_state(tmp_path / "state"))
    sentinel = tmp_path / "state" / "state.json"
    assert sentinel.is_file()
    payload = _json.loads(sentinel.read_text(encoding="utf-8"))
    assert payload["memory_system_id"] == "engram_graph"
    assert payload["engram_manifest"] == "manifest.json"


def test_supports_persistence_is_true() -> None:
    adapter = EngramAdapter(_FakeProvider(), target=_FakeEngram())
    assert adapter.supports_persistence is True


def test_close_closes_the_provider() -> None:
    provider = _FakeProvider()
    adapter = EngramAdapter(provider, target=_FakeEngram())
    asyncio.run(adapter.close())
    assert provider.close_calls == 1


def test_identity_fields_are_read_from_target() -> None:
    adapter = EngramAdapter(_FakeProvider(), target=_FakeEngram())
    assert adapter.memory_system_id == "engram_graph"
    assert adapter.memory_version == "0.4.0"


# ---------------------------- formatter unit ------------------------------


def test_format_recall_context_renders_facts_before_passages() -> None:
    result = _FakeRecallResult(
        passages=(_FakePassage(text="body", node_id="p1"),),
        facts=(_FakeFact(kind="change_event", subject="x", value="y"),),
    )
    rendered = _format_recall_context(result)
    # facts appear first
    assert rendered.index("change event") < rendered.index("body")


def test_format_recall_context_with_no_content_shows_placeholder() -> None:
    assert "no memories" in _format_recall_context(_FakeRecallResult())


def test_passages_to_units_leaves_source_turn_ids_empty() -> None:
    result = _FakeRecallResult(
        passages=(_FakePassage(text="hi", node_id="p1"),),
    )
    units = _passages_to_units(result)
    assert units[0].source_turn_ids == ()


# ---------------------------- factory wiring ------------------------------


def test_factory_resolves_engram_spec(
    fake_engram_module: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch the adapter's lazy engram import path to return our fake.
    import agent_memory_benchmark.adapters.engram_adapter as mod

    class _StubEngram:
        memory_system_id = "engram_graph"
        memory_version = "0.4.0"

    fake_module = sys.modules["engram"]
    fake_module.EngramGraphMemorySystem = _StubEngram  # type: ignore[attr-defined]

    adapter = resolve_adapter("engram", answer_provider=_FakeProvider())
    assert isinstance(adapter, mod.EngramAdapter)
    assert adapter.memory_system_id == "engram_graph"


def test_factory_engram_without_provider_errors() -> None:
    with pytest.raises(AdapterSpecError, match="answer_provider"):
        resolve_adapter("engram")


def test_factory_engram_rejects_http_headers() -> None:
    with pytest.raises(AdapterSpecError, match="--memory-header"):
        resolve_adapter(
            "engram",
            answer_provider=_FakeProvider(),
            http_headers={"Authorization": "x"},
        )


def test_factory_engram_rejects_mappers() -> None:
    with pytest.raises(AdapterSpecError, match="only valid for python adapters"):
        resolve_adapter(
            "engram",
            answer_provider=_FakeProvider(),
            session_mapper=lambda s: s,
        )


def test_factory_engram_forwards_config_kwargs(fake_engram_module: None) -> None:
    """``--memory-config`` kwargs like ``max_passages`` flow to the ctor."""

    import agent_memory_benchmark.adapters.engram_adapter as mod

    class _StubEngram:
        memory_system_id = "engram_graph"
        memory_version = "0.4.0"

    sys.modules["engram"].EngramGraphMemorySystem = _StubEngram  # type: ignore[attr-defined]

    adapter = resolve_adapter(
        "engram",
        answer_provider=_FakeProvider(),
        config={"max_passages": 3},
    )
    assert isinstance(adapter, mod.EngramAdapter)
    assert adapter._max_passages == 3  # noqa: SLF001 (intentional)
