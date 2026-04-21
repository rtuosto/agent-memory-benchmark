"""``EngramAdapter`` — wraps :class:`engram.EngramGraphMemorySystem`.

Engram is a memory *tool* that exposes ``ingest(memory)`` /
``ingest_many(memories)`` and ``recall(query)``. It does not answer
questions — answering is the outside agent's job. This adapter is the
answerer:

1. ``ingest_session`` — build one :class:`engram.Memory` per :class:`Turn`
   and hand the whole session to ``engram.ingest_many``. The graph-backed
   implementation pools spaCy / mpnet / MiniLM forwards across the batch
   dimension (~2.6× faster on synthetic corpora), while preserving the
   ``R16`` append-only ordering and per-memory structural fingerprint of
   a sequential loop.
2. ``answer_question`` — call ``engram.recall(question)``, format the
   :class:`engram.RecallResult` as context, then hand the prompt to the
   user-supplied :class:`LLMProvider` for generation.
3. ``reset`` — delegates to ``engram.reset()``.
4. ``save_state`` / ``load_state`` — delegate to engram's own persistence,
   so the runner can skip re-ingestion on cache hits.

Engram's Python package lives at ``~/code/engram/`` and is ``pip install``-ed
separately. If it's not importable, instantiation fails loudly with guidance
in the exception message.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from dateutil import parser as _dateparser

from ..llm import LLMProvider
from ..types import AnswerResult, RetrievedUnit, Session
from .base import MemoryAdapter

if TYPE_CHECKING:
    from engram import EngramGraphMemorySystem, RecallResult

_log = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = (
    "You are answering a question about a prior conversation. Use only "
    "the memory snippets provided below. If the answer is not supported "
    "by the snippets, say so plainly."
)

_DEFAULT_MAX_PASSAGES = 12


class EngramAdapter(MemoryAdapter):
    """Drive :class:`engram.EngramGraphMemorySystem` from the benchmark runner."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_passages: int = _DEFAULT_MAX_PASSAGES,
        target: EngramGraphMemorySystem | None = None,
    ) -> None:
        if target is None:
            try:
                from engram import EngramGraphMemorySystem as _Engram
            except ImportError as exc:
                raise ImportError(
                    "EngramAdapter requires the 'engram' package to be importable. "
                    "Install it from ~/code/engram/ (pip install -e .) or add it to "
                    "PYTHONPATH."
                ) from exc
            target = _Engram()
        self._target = target
        self._provider = provider
        self._system_prompt = system_prompt
        self._max_passages = max_passages
        self.memory_system_id = target.memory_system_id
        self.memory_version = target.memory_version

    async def ingest_session(self, session: Session, case_id: str) -> None:
        from engram import Memory

        session_ts = _normalize_timestamp(session.session_time)
        memories: list[Memory] = []
        for turn in session.turns:
            metadata: list[tuple[str, str]] = [
                ("case_id", case_id),
                ("session_index", str(session.session_index)),
                ("turn_id", turn.turn_id),
            ]
            if session.session_id:
                metadata.append(("session_id", session.session_id))
            if turn.image_caption:
                metadata.append(("image_caption", turn.image_caption))
            turn_ts = _normalize_timestamp(turn.timestamp) or session_ts
            memories.append(
                Memory(
                    content=turn.text,
                    timestamp=turn_ts,
                    speaker=turn.speaker,
                    source="conversation_turn",
                    metadata=tuple(sorted(metadata)),
                )
            )
        await self._target.ingest_many(memories)

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        t0 = time.perf_counter()
        result = await self._target.recall(question, max_passages=self._max_passages)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        context = _format_recall_context(result)
        user = f"Memory snippets:\n{context}\n\nQuestion: {question}"

        t1 = time.perf_counter()
        chat = await self._provider.chat(system=self._system_prompt, user=user)
        generation_ms = (time.perf_counter() - t1) * 1000.0

        retrieved = _passages_to_units(result)
        tokens_retrieved = sum(len(p.text.split()) for p in result.passages)

        return AnswerResult(
            answer=chat.text,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            units_retrieved=len(result.passages),
            tokens_retrieved=tokens_retrieved,
            retrieved=retrieved,
        )

    async def reset(self) -> None:
        await self._target.reset()

    async def save_state(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        await self._target.save_state(path)
        # The runner's ingestion cache keys on `<dir>/state.json` existing.
        # Engram's native layout writes manifest.json + primary.msgpack +
        # embeddings.npy + node_ids.json — no state.json — so without a
        # sentinel the runner can't detect a warm cache and always re-ingests.
        # Write a minimal pointer so cache hits fire; load_state still
        # delegates to engram.load_state on the directory.
        sentinel = path / "state.json"
        sentinel.write_text(
            json.dumps(
                {
                    "memory_system_id": self.memory_system_id,
                    "memory_version": self.memory_version,
                    "engram_manifest": "manifest.json",
                }
            ),
            encoding="utf-8",
        )

    async def load_state(self, path: Path) -> None:
        await self._target.load_state(path)

    @property
    def supports_persistence(self) -> bool:
        return True

    async def close(self) -> None:
        await self._provider.close()


def _format_recall_context(result: RecallResult) -> str:
    """Render a :class:`RecallResult` into a compact prompt-ready block.

    Facts first (they are pre-computed answers the agent can cite
    verbatim), then ranked passages with speaker / timestamp prefixes
    when engram populated them.
    """

    lines: list[str] = []
    if result.facts:
        lines.append("Relevant facts:")
        for fact in result.facts:
            label = fact.kind.replace("_", " ")
            lines.append(f"- ({label}) {fact.subject}: {fact.value}")
        lines.append("")
    if result.passages:
        lines.append("Ranked memory passages:")
        for idx, passage in enumerate(result.passages, start=1):
            prefix_parts: list[str] = []
            if passage.speaker:
                prefix_parts.append(passage.speaker)
            if passage.timestamp:
                prefix_parts.append(passage.timestamp)
            prefix = f" [{' · '.join(prefix_parts)}]" if prefix_parts else ""
            lines.append(f"{idx}.{prefix} {passage.text}")
    if not lines:
        lines.append("(no memories retrieved)")
    return "\n".join(lines).rstrip()


def _normalize_timestamp(raw: str | None) -> str | None:
    """Coerce an arbitrary timestamp string to ISO-8601, or ``None``.

    Engram resolves timestamps with :func:`datetime.datetime.fromisoformat`,
    which rejects the LongMemEval format (``"2023/05/20 (Sat) 02:21"``) and
    anything else that isn't strict ISO. We try :mod:`dateutil.parser` for a
    best-effort conversion; on failure we drop the timestamp rather than
    blowing up ingest — engram treats ``None`` as "only relative ordering
    available," which is the correct fallback for un-parseable input.
    """

    if raw is None or not raw.strip():
        return None
    try:
        parsed = _dateparser.parse(raw, fuzzy=True)
    except (ValueError, OverflowError, TypeError) as exc:
        _log.debug("dropping unparseable timestamp %r: %s", raw, exc)
        return None
    return str(parsed.isoformat())


def _passages_to_units(result: RecallResult) -> tuple[RetrievedUnit, ...]:
    """Project engram's :class:`RecallPassage`es into benchmark units.

    ``unit_id`` comes from the passage's graph node id. ``source_turn_ids``
    is left empty — engram doesn't currently expose the original turn id
    via the passage projection; the scorer falls back to substring
    matching against known turn text for evidence KPIs.
    """

    return tuple(
        RetrievedUnit(
            unit_id=passage.node_id,
            text=passage.text,
            score=passage.score,
            source_turn_ids=(),
        )
        for passage in result.passages
    )


__all__ = ["EngramAdapter"]
