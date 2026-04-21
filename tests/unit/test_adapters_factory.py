"""Tests for :func:`resolve_adapter`."""

from __future__ import annotations

import sys
import types

import pytest

from agent_memory_benchmark.adapters.factory import AdapterSpecError, resolve_adapter
from agent_memory_benchmark.adapters.full_context import FullContextAdapter
from agent_memory_benchmark.adapters.python_adapter import PythonAdapter
from agent_memory_benchmark.llm import ChatResult
from agent_memory_benchmark.types import AnswerResult, Session


class _FakeProvider:
    spec = "fake:model"
    model = "model"

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        return ChatResult(text="", model=self.model)

    async def resolve_spec(self) -> str:
        return self.spec

    async def close(self) -> None:
        return None


class _GoodTarget:
    memory_system_id = "good"
    memory_version = "1.0"

    async def ingest_session(self, session: Session, case_id: str) -> None:
        return None

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        return AnswerResult(answer="", retrieval_time_ms=0.0, generation_time_ms=0.0)

    async def reset(self) -> None:
        return None


_MODULE = "_amb_factory_test_module"


@pytest.fixture
def plantable_module() -> types.ModuleType:
    module = types.ModuleType(_MODULE)
    module.GoodTarget = _GoodTarget  # type: ignore[attr-defined]
    sys.modules[_MODULE] = module
    try:
        yield module
    finally:
        sys.modules.pop(_MODULE, None)


def test_full_context_returns_full_context_adapter() -> None:
    adapter = resolve_adapter("full-context", answer_provider=_FakeProvider())
    assert isinstance(adapter, FullContextAdapter)


def test_full_context_without_provider_errors() -> None:
    with pytest.raises(AdapterSpecError, match="answer_provider"):
        resolve_adapter("full-context")


def test_python_spec_returns_python_adapter(
    plantable_module: types.ModuleType,
) -> None:
    adapter = resolve_adapter(f"python:{_MODULE}:GoodTarget")
    assert isinstance(adapter, PythonAdapter)
    assert adapter.memory_system_id == "good"


def test_python_spec_missing_target_errors() -> None:
    with pytest.raises(AdapterSpecError, match="python adapter needs a target"):
        resolve_adapter("python:")


def test_http_is_reserved_for_pr10() -> None:
    with pytest.raises(AdapterSpecError, match="PR-10"):
        resolve_adapter("http://localhost:8000")


@pytest.mark.parametrize("bad", ["", "   ", "   :   "])
def test_empty_or_kindless_spec_errors(bad: str) -> None:
    with pytest.raises(AdapterSpecError):
        resolve_adapter(bad)


def test_unknown_kind_errors() -> None:
    with pytest.raises(AdapterSpecError, match="Unknown adapter kind"):
        resolve_adapter("subprocess:whatever")


def test_kind_is_case_insensitive(plantable_module: types.ModuleType) -> None:
    adapter = resolve_adapter(f"Python:{_MODULE}:GoodTarget")
    assert isinstance(adapter, PythonAdapter)


def test_full_context_colon_form_also_works() -> None:
    """``full-context:`` (trailing colon) should resolve the same as bare ``full-context``."""

    adapter = resolve_adapter("full-context:", answer_provider=_FakeProvider())
    assert isinstance(adapter, FullContextAdapter)
