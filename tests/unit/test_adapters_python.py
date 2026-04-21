"""Tests for :class:`PythonAdapter`."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from agent_memory_benchmark.adapters.python_adapter import (
    PythonAdapter,
    PythonAdapterError,
)
from agent_memory_benchmark.types import AnswerResult, Session, Turn

# --- Test targets --------------------------------------------------------


class _GoodTarget:
    memory_system_id = "good"
    memory_version = "1.0"

    def __init__(self, label: str = "default") -> None:
        self.label = label
        self.ingested: list[tuple[Session, str]] = []
        self.question_log: list[tuple[str, str]] = []
        self.reset_count = 0
        self.closed = False

    async def ingest_session(self, session: Session, case_id: str) -> None:
        self.ingested.append((session, case_id))

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        self.question_log.append((question, case_id))
        return AnswerResult(
            answer=f"{self.label}:{question}",
            retrieval_time_ms=1.0,
            generation_time_ms=2.0,
            units_retrieved=len(self.ingested),
            tokens_retrieved=5,
            retrieved=(),
        )

    async def reset(self) -> None:
        self.ingested.clear()
        self.reset_count += 1

    async def save_state(self, path: Path) -> None:
        (path / "state.json").write_text(f"{len(self.ingested)}", encoding="utf-8")

    async def load_state(self, path: Path) -> None:
        (path / "state.json").read_text(encoding="utf-8")

    async def close(self) -> None:
        self.closed = True


class _TargetWithoutPersistence:
    memory_system_id = "nopersist"
    memory_version = "1.0"

    async def ingest_session(self, session: Session, case_id: str) -> None:
        return None

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        return AnswerResult(answer="", retrieval_time_ms=0.0, generation_time_ms=0.0)

    async def reset(self) -> None:
        return None


class _TargetMissingMethods:
    memory_system_id = "missing"
    memory_version = "1.0"

    # Intentionally missing answer_question / reset.
    async def ingest_session(self, session: Session, case_id: str) -> None:
        return None


class _TargetReturningForeignType:
    memory_system_id = "foreign"
    memory_version = "1.0"

    async def ingest_session(self, session: Session, case_id: str) -> None:
        return None

    async def answer_question(self, question: str, case_id: str) -> Any:
        return {"answer": "wrong type"}

    async def reset(self) -> None:
        return None


# --- Module fixture for from_spec ---------------------------------------


_TEST_MODULE_NAME = "_amb_test_targets_module"


@pytest.fixture
def plantable_module() -> types.ModuleType:
    """Plant a synthetic module into sys.modules so from_spec can find it."""

    module = types.ModuleType(_TEST_MODULE_NAME)
    module.GoodTarget = _GoodTarget  # type: ignore[attr-defined]
    module.BadTarget = _TargetMissingMethods  # type: ignore[attr-defined]
    module.NoPersist = _TargetWithoutPersistence  # type: ignore[attr-defined]
    module.Foreign = _TargetReturningForeignType  # type: ignore[attr-defined]
    sys.modules[_TEST_MODULE_NAME] = module
    try:
        yield module
    finally:
        sys.modules.pop(_TEST_MODULE_NAME, None)


# --- Direct construction -------------------------------------------------


def test_identity_fields_mirror_target() -> None:
    target = _GoodTarget()
    adapter = PythonAdapter(target)
    assert adapter.memory_system_id == target.memory_system_id == "good"
    assert adapter.memory_version == target.memory_version == "1.0"


async def test_delegates_ingest_answer_reset() -> None:
    target = _GoodTarget(label="hello")
    adapter = PythonAdapter(target)
    session = Session(
        session_index=1,
        turns=(Turn(turn_id="1:0", speaker="alice", text="hi"),),
    )
    await adapter.ingest_session(session, "c1")
    assert target.ingested == [(session, "c1")]

    result = await adapter.answer_question("q?", "c1")
    assert isinstance(result, AnswerResult)
    assert result.answer == "hello:q?"

    await adapter.reset()
    assert target.reset_count == 1
    assert target.ingested == []


async def test_session_mapper_is_applied() -> None:
    target = _GoodTarget()
    seen: list[object] = []

    async def _spy_ingest(session: object, case_id: str) -> None:
        seen.append(session)

    target.ingest_session = _spy_ingest  # type: ignore[assignment]

    def mapper(session: Session) -> dict[str, object]:
        return {"mapped": True, "turns": len(session.turns)}

    adapter = PythonAdapter(target, session_mapper=mapper)
    await adapter.ingest_session(Session(session_index=1, turns=()), "c")
    assert seen == [{"mapped": True, "turns": 0}]


async def test_result_mapper_converts_foreign_return() -> None:
    def mapper(raw: object) -> AnswerResult:
        assert isinstance(raw, dict)
        return AnswerResult(
            answer=str(raw["answer"]),
            retrieval_time_ms=0.0,
            generation_time_ms=0.0,
        )

    adapter = PythonAdapter(_TargetReturningForeignType(), result_mapper=mapper)
    result = await adapter.answer_question("q", "c")
    assert result.answer == "wrong type"


async def test_foreign_return_without_mapper_raises() -> None:
    adapter = PythonAdapter(_TargetReturningForeignType())
    with pytest.raises(PythonAdapterError, match="AnswerResult"):
        await adapter.answer_question("q", "c")


# --- from_spec -----------------------------------------------------------


def test_from_spec_loads_and_instantiates(plantable_module: types.ModuleType) -> None:
    adapter = PythonAdapter.from_spec(
        f"{_TEST_MODULE_NAME}:GoodTarget", config={"label": "configured"}
    )
    assert isinstance(adapter._target, _GoodTarget)  # type: ignore[attr-defined]
    assert adapter._target.label == "configured"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "spec",
    ["bad", ":Class", "pkg:", "   "],
)
def test_from_spec_rejects_malformed(spec: str) -> None:
    with pytest.raises(PythonAdapterError):
        PythonAdapter.from_spec(spec)


def test_from_spec_unknown_module() -> None:
    with pytest.raises(PythonAdapterError, match="import"):
        PythonAdapter.from_spec("definitely_not_a_real_module_abc:Class")


def test_from_spec_unknown_class(plantable_module: types.ModuleType) -> None:
    with pytest.raises(PythonAdapterError, match="no attribute"):
        PythonAdapter.from_spec(f"{_TEST_MODULE_NAME}:NoSuchClass")


def test_from_spec_rejects_bad_kwargs(plantable_module: types.ModuleType) -> None:
    with pytest.raises(PythonAdapterError, match="instantiate"):
        PythonAdapter.from_spec(
            f"{_TEST_MODULE_NAME}:GoodTarget",
            config={"not_a_real_arg": 1},
        )


def test_from_spec_rejects_non_conforming(plantable_module: types.ModuleType) -> None:
    with pytest.raises(PythonAdapterError, match="MemorySystemShape"):
        PythonAdapter.from_spec(f"{_TEST_MODULE_NAME}:BadTarget")


# --- Persistence delegation ---------------------------------------------


async def test_supports_persistence_mirrors_target() -> None:
    assert PythonAdapter(_GoodTarget()).supports_persistence is True
    assert PythonAdapter(_TargetWithoutPersistence()).supports_persistence is False


async def test_save_load_delegate_when_target_supports(tmp_path: Path) -> None:
    adapter = PythonAdapter(_GoodTarget())
    await adapter.save_state(tmp_path)
    assert (tmp_path / "state.json").is_file()
    await adapter.load_state(tmp_path)


async def test_save_raises_when_target_lacks_persistence(tmp_path: Path) -> None:
    adapter = PythonAdapter(_TargetWithoutPersistence())
    with pytest.raises(NotImplementedError):
        await adapter.save_state(tmp_path)
    with pytest.raises(NotImplementedError):
        await adapter.load_state(tmp_path)


async def test_close_delegates_when_target_has_close() -> None:
    target = _GoodTarget()
    adapter = PythonAdapter(target)
    await adapter.close()
    assert target.closed is True


async def test_close_is_noop_when_target_has_no_close() -> None:
    target = _TargetWithoutPersistence()
    adapter = PythonAdapter(target)
    await adapter.close()  # must not raise
