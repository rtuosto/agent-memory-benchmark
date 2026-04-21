"""Tests for :class:`FullContextAdapter`."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_memory_benchmark.adapters.full_context import FullContextAdapter
from agent_memory_benchmark.llm import ChatResult
from agent_memory_benchmark.types import Session, Turn


class _RecordingProvider:
    spec = "fake:model"
    model = "model"

    def __init__(self, reply: str = "answer") -> None:
        self._reply = reply
        self.calls: list[dict[str, object]] = []
        self.closed = False

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        return ChatResult(text=self._reply, model=self.model)

    async def resolve_spec(self) -> str:
        return self.spec

    async def close(self) -> None:
        self.closed = True


def _session(idx: int, *turns: tuple[str, str, str | None]) -> Session:
    return Session(
        session_index=idx,
        turns=tuple(
            Turn(turn_id=f"{idx}:{i}", speaker=sp, text=txt, timestamp=ts)
            for i, (sp, txt, ts) in enumerate(turns)
        ),
        session_time=f"2026-04-{idx:02d}T00:00:00",
    )


async def test_identity_fields() -> None:
    adapter = FullContextAdapter(_RecordingProvider())
    assert adapter.memory_system_id == "full-context"
    assert adapter.memory_version == "0.1.0"


async def test_answer_concatenates_all_ingested_turns() -> None:
    provider = _RecordingProvider(reply="42")
    adapter = FullContextAdapter(provider)
    await adapter.ingest_session(_session(1, ("alice", "I bought a kayak.", None)), "case_a")
    await adapter.ingest_session(
        _session(2, ("alice", "It's blue.", None), ("bob", "Nice!", None)),
        "case_a",
    )

    result = await adapter.answer_question("What did alice buy?", "case_a")

    assert result.answer == "42"
    assert result.retrieval_time_ms == 0.0
    assert result.generation_time_ms >= 0.0
    assert result.units_retrieved == 3  # three turns total
    assert result.tokens_retrieved > 0
    assert result.retrieved == ()

    assert len(provider.calls) == 1
    user = provider.calls[0]["user"]
    assert isinstance(user, str)
    assert "alice: I bought a kayak." in user
    assert "bob: Nice!" in user
    assert "What did alice buy?" in user


async def test_answer_isolates_by_case_id() -> None:
    provider = _RecordingProvider()
    adapter = FullContextAdapter(provider)
    await adapter.ingest_session(_session(1, ("a", "case A content", None)), "A")
    await adapter.ingest_session(_session(1, ("a", "case B content", None)), "B")

    await adapter.answer_question("q", "A")
    user_a = provider.calls[-1]["user"]
    assert isinstance(user_a, str)
    assert "case A content" in user_a
    assert "case B content" not in user_a


async def test_timestamp_is_rendered_when_present() -> None:
    provider = _RecordingProvider()
    adapter = FullContextAdapter(provider)
    await adapter.ingest_session(_session(1, ("alice", "hi", "2026-04-20T12:00")), "c")
    await adapter.answer_question("q", "c")
    user = provider.calls[0]["user"]
    assert isinstance(user, str)
    assert "alice (2026-04-20T12:00): hi" in user


async def test_reset_clears_state() -> None:
    provider = _RecordingProvider()
    adapter = FullContextAdapter(provider)
    await adapter.ingest_session(_session(1, ("a", "something", None)), "c")
    await adapter.reset()
    await adapter.answer_question("q", "c")
    user = provider.calls[0]["user"]
    assert isinstance(user, str)
    assert "something" not in user
    assert user.startswith("Conversation transcript:")


async def test_save_and_load_state_roundtrip(tmp_path: Path) -> None:
    provider = _RecordingProvider()
    adapter = FullContextAdapter(provider)
    await adapter.ingest_session(_session(1, ("alice", "stored", "2026-04-20T00:00")), "cA")
    await adapter.ingest_session(_session(2, ("bob", "also stored", None)), "cA")
    await adapter.save_state(tmp_path)
    assert (tmp_path / "state.json").is_file()

    fresh = FullContextAdapter(_RecordingProvider())
    await fresh.load_state(tmp_path)
    await fresh.answer_question("q", "cA")
    restored_provider = fresh._provider  # type: ignore[attr-defined]
    assert isinstance(restored_provider, _RecordingProvider)
    user = restored_provider.calls[0]["user"]
    assert isinstance(user, str)
    assert "alice (2026-04-20T00:00): stored" in user
    assert "bob: also stored" in user


def test_supports_persistence_true() -> None:
    assert FullContextAdapter(_RecordingProvider()).supports_persistence is True


async def test_close_closes_provider() -> None:
    provider = _RecordingProvider()
    adapter = FullContextAdapter(provider)
    await adapter.close()
    assert provider.closed is True


async def test_custom_system_prompt_is_forwarded() -> None:
    provider = _RecordingProvider()
    adapter = FullContextAdapter(provider, system_prompt="custom prompt")
    await adapter.answer_question("q", "any")
    assert provider.calls[0]["system"] == "custom prompt"


async def test_missing_case_produces_empty_transcript() -> None:
    """answer_question on an unknown case_id must still produce a valid call."""

    provider = _RecordingProvider(reply="idk")
    adapter = FullContextAdapter(provider)
    result = await adapter.answer_question("q", "never-ingested")
    assert result.units_retrieved == 0
    user = provider.calls[0]["user"]
    assert isinstance(user, str)
    assert "q" in user


async def test_load_state_rejects_missing_file(tmp_path: Path) -> None:
    adapter = FullContextAdapter(_RecordingProvider())
    with pytest.raises(FileNotFoundError):
        await adapter.load_state(tmp_path)
