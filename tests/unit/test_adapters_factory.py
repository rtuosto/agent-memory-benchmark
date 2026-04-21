"""Tests for :func:`resolve_adapter`."""

from __future__ import annotations

import sys
import types

import pytest

from agent_memory_benchmark.adapters.factory import AdapterSpecError, resolve_adapter
from agent_memory_benchmark.adapters.full_context import FullContextAdapter
from agent_memory_benchmark.adapters.http_adapter import HttpAdapter
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


def test_http_spec_returns_http_adapter() -> None:
    adapter = resolve_adapter("http://localhost:8000")
    assert isinstance(adapter, HttpAdapter)


def test_https_spec_returns_http_adapter() -> None:
    adapter = resolve_adapter("https://mem.example.com:443")
    assert isinstance(adapter, HttpAdapter)


def test_http_spec_forwards_headers() -> None:
    headers = {"Authorization": "Bearer TOKEN"}
    adapter = resolve_adapter("http://localhost:8000", http_headers=headers)
    assert isinstance(adapter, HttpAdapter)
    # Internal state check is acceptable here — the factory has no other
    # observable surface for headers until the adapter is opened.
    assert adapter._headers == headers  # noqa: SLF001 (intentional)


def test_http_spec_rejects_python_config() -> None:
    with pytest.raises(AdapterSpecError, match="only valid for python adapters"):
        resolve_adapter("http://localhost:8000", config={"foo": "bar"})


def test_http_spec_rejects_mappers() -> None:
    with pytest.raises(AdapterSpecError, match="only valid for python adapters"):
        resolve_adapter("http://localhost:8000", session_mapper=lambda s: s)


def test_non_http_spec_rejects_http_headers(
    plantable_module: types.ModuleType,
) -> None:
    with pytest.raises(AdapterSpecError, match="--memory-header"):
        resolve_adapter(
            f"python:{_MODULE}:GoodTarget",
            http_headers={"Authorization": "x"},
        )
    with pytest.raises(AdapterSpecError, match="--memory-header"):
        resolve_adapter(
            "full-context",
            answer_provider=_FakeProvider(),
            http_headers={"Authorization": "x"},
        )


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


def test_python_spec_forwards_mappers_to_python_adapter(
    plantable_module: types.ModuleType,
) -> None:
    """`resolve_adapter` must pass `session_mapper` / `result_mapper` through."""

    sessions_seen: list[Session] = []

    def session_mapper(s: Session) -> Session:
        sessions_seen.append(s)
        return s

    def result_mapper(raw: object) -> AnswerResult:
        # Accept anything; the real adapter only calls this on answer_question.
        return AnswerResult(answer="mapped", retrieval_time_ms=0.0, generation_time_ms=0.0)

    adapter = resolve_adapter(
        f"python:{_MODULE}:GoodTarget",
        session_mapper=session_mapper,
        result_mapper=result_mapper,
    )
    assert isinstance(adapter, PythonAdapter)
    assert adapter._session_mapper is session_mapper  # noqa: SLF001 (intentional)
    assert adapter._result_mapper is result_mapper  # noqa: SLF001


def test_full_context_rejects_mapper_kwargs() -> None:
    """Mappers only apply to python adapters; passing them elsewhere is a misuse."""

    with pytest.raises(AdapterSpecError, match="only valid for python adapters"):
        resolve_adapter(
            "full-context",
            answer_provider=_FakeProvider(),
            session_mapper=lambda s: s,
        )
