"""Tests for :class:`LongMemEvalJudge` dispatch + fingerprint selection."""

from __future__ import annotations

import asyncio

import pytest

from agent_memory_benchmark.judge.longmemeval import LME_JUDGE_FINGERPRINT, LME_PROMPT_FINGERPRINTS
from agent_memory_benchmark.llm import ChatResult
from agent_memory_benchmark.llm.judge_client import JudgeClient
from agent_memory_benchmark.runner.judge_adapter import LongMemEvalJudge, _template_key_for
from agent_memory_benchmark.types import QAItem


class _FakeProvider:
    def __init__(self, response: str) -> None:
        self._response = response
        self.model = "judge"
        self.spec = "fake:judge"
        self.last_prompt: str | None = None

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        self.last_prompt = user
        return ChatResult(text=self._response, model=self.model)

    async def resolve_spec(self) -> str:
        return self.spec

    async def close(self) -> None:
        return None


def _judge(response: str = "yes") -> tuple[LongMemEvalJudge, _FakeProvider]:
    provider = _FakeProvider(response)
    client = JudgeClient(provider, temperature=0.0)
    judge = LongMemEvalJudge(
        client,
        runs=1,
        temperature=0.0,
        bundle_fingerprint=LME_JUDGE_FINGERPRINT,
    )
    return judge, provider


def _qa(*, question_type: str = "single-session-user", question_id: str = "q1") -> QAItem:
    return QAItem(
        question_id=question_id,
        question="Q?",
        gold="A",
        question_type=question_type,
    )


def test_judge_parses_yes_response() -> None:
    judge, _ = _judge("yes")
    outcome = asyncio.run(judge.judge(_qa(), "some generated text"))
    assert outcome.verdicts == [{"correct": True, "raw": "yes"}]
    assert outcome.prompt_fingerprint == LME_PROMPT_FINGERPRINTS["general"]


def test_judge_parses_no_response() -> None:
    judge, _ = _judge("no")
    outcome = asyncio.run(judge.judge(_qa(), "wrong"))
    assert outcome.verdicts == [{"correct": False, "raw": "no"}]


def test_judge_routes_abstention_by_question_id() -> None:
    judge, _ = _judge("yes")
    outcome = asyncio.run(judge.judge(_qa(question_id="q1_abs_42"), "I don't know"))
    assert outcome.prompt_fingerprint == LME_PROMPT_FINGERPRINTS["abstention"]


def test_judge_selects_temporal_template() -> None:
    judge, _ = _judge("yes")
    outcome = asyncio.run(judge.judge(_qa(question_type="temporal-reasoning"), "19 days"))
    assert outcome.prompt_fingerprint == LME_PROMPT_FINGERPRINTS["temporal-reasoning"]


def test_judge_selects_knowledge_update_template() -> None:
    judge, _ = _judge("yes")
    outcome = asyncio.run(judge.judge(_qa(question_type="knowledge-update"), "updated!"))
    assert outcome.prompt_fingerprint == LME_PROMPT_FINGERPRINTS["knowledge-update"]


def test_judge_selects_preference_template() -> None:
    judge, _ = _judge("yes")
    outcome = asyncio.run(judge.judge(_qa(question_type="single-session-preference"), "pref"))
    assert outcome.prompt_fingerprint == LME_PROMPT_FINGERPRINTS["single-session-preference"]


def test_judge_rejects_multi_run_for_longmemeval() -> None:
    provider = _FakeProvider("yes")
    client = JudgeClient(provider, temperature=0.0)
    with pytest.raises(ValueError, match="supports --judge-runs 1"):
        LongMemEvalJudge(client, runs=3, temperature=0.0, bundle_fingerprint=LME_JUDGE_FINGERPRINT)


def test_template_key_for_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unsupported LongMemEval question_type"):
        _template_key_for("unknown-task", abstention=False)


def test_judge_reports_time() -> None:
    judge, _ = _judge("yes")
    outcome = asyncio.run(judge.judge(_qa(), "gen"))
    assert outcome.judge_time_ms >= 0.0
