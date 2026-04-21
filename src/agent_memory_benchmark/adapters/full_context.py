"""``FullContextAdapter`` — the "null memory" baseline.

Concatenates every ingested turn into the prompt at answer time. No
retrieval is performed; ``units_retrieved`` reflects the total turn count
and ``tokens_retrieved`` is a whitespace approximation of context size.
The runner uses this as a smoke test and as a reference upper-bound on
what any real memory system should at least match when the full context
fits in the answer model's window.

Persistence is implemented: state is a ``{case_id -> [Session, ...]}``
dict which JSON-serializes trivially. That lets the runner exercise the
ingestion cache path even for the baseline.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..llm import LLMProvider
from ..types import AnswerResult, Session, Turn
from .base import MemoryAdapter

_DEFAULT_SYSTEM_PROMPT = (
    "You are answering questions about a prior conversation. Use only the "
    "conversation transcript below to answer. If the answer is not present, "
    "say so plainly."
)


class FullContextAdapter(MemoryAdapter):
    """Null-memory baseline: feeds the entire conversation back to the LLM."""

    memory_system_id = "full-context"
    memory_version = "0.1.0"

    def __init__(
        self,
        provider: LLMProvider,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt
        self._sessions: dict[str, list[Session]] = {}

    async def ingest_session(self, session: Session, case_id: str) -> None:
        self._sessions.setdefault(case_id, []).append(session)

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        context_text, turn_count, token_est = self._build_context(case_id)
        user = f"Conversation transcript:\n{context_text}\n\nQuestion: {question}"

        t0 = time.perf_counter()
        result = await self._provider.chat(system=self._system_prompt, user=user)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return AnswerResult(
            answer=result.text,
            retrieval_time_ms=0.0,
            generation_time_ms=elapsed_ms,
            units_retrieved=turn_count,
            tokens_retrieved=token_est,
            retrieved=(),
        )

    async def reset(self) -> None:
        self._sessions.clear()

    async def save_state(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "memory_system_id": self.memory_system_id,
            "memory_version": self.memory_version,
            "sessions": {
                case_id: [asdict(session) for session in sessions]
                for case_id, sessions in self._sessions.items()
            },
        }
        (path / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    async def load_state(self, path: Path) -> None:
        state_file = path / "state.json"
        payload = json.loads(state_file.read_text(encoding="utf-8"))
        self._sessions = {}
        for case_id, raw_sessions in payload.get("sessions", {}).items():
            sessions: list[Session] = []
            for raw in raw_sessions:
                turns = tuple(Turn(**t) for t in raw.get("turns", ()))
                sessions.append(
                    Session(
                        session_index=raw["session_index"],
                        turns=turns,
                        session_time=raw.get("session_time"),
                        session_id=raw.get("session_id"),
                    )
                )
            self._sessions[case_id] = sessions

    async def close(self) -> None:
        await self._provider.close()

    def _build_context(self, case_id: str) -> tuple[str, int, int]:
        sessions = self._sessions.get(case_id, [])
        lines: list[str] = []
        turn_count = 0
        for session in sessions:
            if session.session_time:
                lines.append(f"[Session {session.session_index} — {session.session_time}]")
            else:
                lines.append(f"[Session {session.session_index}]")
            for turn in session.turns:
                prefix = f"{turn.speaker}:"
                if turn.timestamp:
                    prefix = f"{turn.speaker} ({turn.timestamp}):"
                lines.append(f"{prefix} {turn.text}")
                turn_count += 1
            lines.append("")
        context = "\n".join(lines).rstrip()
        # Whitespace-split token estimate; callers with real tokenizers can
        # substitute a better measure at the scorer layer.
        token_est = len(context.split())
        return context, turn_count, token_est


__all__ = ["FullContextAdapter"]
