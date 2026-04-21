"""``PythonAdapter`` — in-process memory system loader.

Loads a class given as ``"pkg.module:ClassName"``, instantiates it with
user-supplied kwargs, and structurally checks the instance against
:class:`agent_memory_benchmark.compat.MemorySystemShape`. Benchmark types
pass through unchanged by default. Memory systems that use their own
``Session``/``AnswerResult`` types can supply mapper callables via
``--memory-config`` (or the constructor's ``session_mapper`` /
``result_mapper`` kwargs) to translate at the boundary.

Persistence delegates to the target iff it exposes async ``save_state`` /
``load_state``; otherwise the methods raise :class:`NotImplementedError`
and the runner falls back to re-ingestion on every run.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..compat import MemorySystemShape
from ..types import AnswerResult, Session
from .base import MemoryAdapter

SessionMapper = Callable[[Session], Any]
ResultMapper = Callable[[Any], AnswerResult]


class PythonAdapterError(Exception):
    """Raised for configuration or structural-type problems."""


class PythonAdapter(MemoryAdapter):
    """Wraps a :class:`MemorySystemShape`-compatible in-process target.

    ``session_mapper`` (optional) converts the benchmark's :class:`Session`
    into whatever type the target's ``ingest_session`` expects — identity
    by default. ``result_mapper`` (optional) converts the target's answer
    return value into :class:`AnswerResult`. Used by adapter callers that
    bridge a target with divergent type names or shapes.
    """

    def __init__(
        self,
        target: MemorySystemShape,
        *,
        session_mapper: SessionMapper | None = None,
        result_mapper: ResultMapper | None = None,
    ) -> None:
        self._target = target
        self._session_mapper = session_mapper
        self._result_mapper = result_mapper
        self.memory_system_id = target.memory_system_id
        self.memory_version = target.memory_version

    @classmethod
    def from_spec(
        cls,
        target_spec: str,
        *,
        config: dict[str, Any] | None = None,
        session_mapper: SessionMapper | None = None,
        result_mapper: ResultMapper | None = None,
    ) -> PythonAdapter:
        """Construct from ``"pkg.module:ClassName"`` + ctor kwargs.

        Raises :class:`PythonAdapterError` if the spec is malformed, the
        module or class cannot be resolved, or the instantiated object
        does not match :class:`MemorySystemShape`.
        """

        module_path, _, class_name = target_spec.partition(":")
        if not module_path or not class_name:
            raise PythonAdapterError(
                f"Invalid python adapter target {target_spec!r}; expected 'pkg.module:ClassName'."
            )
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise PythonAdapterError(f"Could not import {module_path!r}: {exc}") from exc
        try:
            klass = getattr(module, class_name)
        except AttributeError as exc:
            raise PythonAdapterError(
                f"Module {module_path!r} has no attribute {class_name!r}."
            ) from exc
        try:
            instance = klass(**(config or {}))
        except TypeError as exc:
            raise PythonAdapterError(
                f"Could not instantiate {target_spec!r} with config={config!r}: {exc}"
            ) from exc

        if not isinstance(instance, MemorySystemShape):
            missing = _describe_missing_shape(instance)
            raise PythonAdapterError(
                f"{target_spec!r} does not match MemorySystemShape. "
                f"Missing or mistyped: {missing}. See docs/compat.md."
            )
        return cls(
            instance,
            session_mapper=session_mapper,
            result_mapper=result_mapper,
        )

    async def ingest_session(self, session: Session, case_id: str) -> None:
        mapped = self._session_mapper(session) if self._session_mapper else session
        await self._target.ingest_session(mapped, case_id)

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        raw = await self._target.answer_question(question, case_id)
        if self._result_mapper is not None:
            return self._result_mapper(raw)
        if not isinstance(raw, AnswerResult):
            raise PythonAdapterError(
                f"{type(self._target).__name__}.answer_question returned "
                f"{type(raw).__name__}, expected AnswerResult (or supply "
                "result_mapper)."
            )
        return raw

    async def reset(self) -> None:
        await self._target.reset()

    async def save_state(self, path: Path) -> None:
        save = getattr(self._target, "save_state", None)
        if save is None:
            raise NotImplementedError(f"{type(self._target).__name__} does not expose save_state.")
        await save(path)

    async def load_state(self, path: Path) -> None:
        load = getattr(self._target, "load_state", None)
        if load is None:
            raise NotImplementedError(f"{type(self._target).__name__} does not expose load_state.")
        await load(path)

    @property
    def supports_persistence(self) -> bool:
        return hasattr(self._target, "save_state") and hasattr(self._target, "load_state")

    async def close(self) -> None:
        close = getattr(self._target, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


def _describe_missing_shape(instance: object) -> str:
    """Best-effort list of what a target is missing vs. MemorySystemShape.

    Used to make the runtime-check error actionable rather than just
    reporting ``isinstance(..., MemorySystemShape) is False``.
    """

    required_attrs = ("memory_system_id", "memory_version")
    required_methods = ("ingest_session", "answer_question", "reset")
    problems: list[str] = []
    for name in required_attrs:
        if not hasattr(instance, name):
            problems.append(f"attribute {name!r}")
    for name in required_methods:
        method = getattr(instance, name, None)
        if method is None:
            problems.append(f"method {name!r}")
        elif not callable(method):
            problems.append(f"{name!r} is not callable")
    return ", ".join(problems) if problems else "<unknown>"


__all__ = ["PythonAdapter", "PythonAdapterError", "ResultMapper", "SessionMapper"]
