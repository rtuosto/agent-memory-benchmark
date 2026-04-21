"""Tests for :class:`LongMemEvalJudge` dispatch + fingerprint selection."""

from __future__ import annotations

import asyncio

import pytest

from agent_memory_benchmark.judge.beam import (
    BEAM_JUDGE_FINGERPRINT,
    BEAM_PROMPT_FINGERPRINTS,
)
from agent_memory_benchmark.judge.locomo import (
    LOCOMO_JUDGE_FINGERPRINT,
    LOCOMO_PROMPT_FINGERPRINTS,
)
from agent_memory_benchmark.judge.longmemeval import LME_JUDGE_FINGERPRINT, LME_PROMPT_FINGERPRINTS
from agent_memory_benchmark.llm import ChatResult
from agent_memory_benchmark.llm.judge_client import JudgeClient
from agent_memory_benchmark.runner.judge_adapter import (
    BeamJudge,
    LocomoJudge,
    LongMemEvalJudge,
    _template_key_for,
    locomo_majority_correct,
)
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


def test_longmemeval_prompt_fingerprint_matches_judge_write() -> None:
    """``prompt_fingerprint(qa)`` must equal the fp the judge embeds in its
    :class:`JudgeOutcome`. Orchestrator cache lookup depends on this."""

    judge, _ = _judge("yes")
    qa = _qa(question_type="temporal-reasoning")
    outcome = asyncio.run(judge.judge(qa, "gen"))
    assert judge.prompt_fingerprint(qa) == outcome.prompt_fingerprint


class _MultiFakeProvider:
    """Records each chat() call; returns canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []
        self.model = "judge"
        self.spec = "fake:judge"

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        self.calls.append(user)
        # Always return the first response when we have fewer distinct
        # responses than calls — sized-1 lists act as "constant reply".
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return ChatResult(text=self._responses[idx], model=self.model)

    async def resolve_spec(self) -> str:
        return self.spec

    async def close(self) -> None:
        return None


def _locomo_judge(responses: list[str], *, runs: int = 3) -> tuple[LocomoJudge, _MultiFakeProvider]:
    provider = _MultiFakeProvider(responses)
    client = JudgeClient(provider, temperature=0.0)
    judge = LocomoJudge(
        client,
        runs=runs,
        temperature=0.0,
        bundle_fingerprint=LOCOMO_JUDGE_FINGERPRINT,
    )
    return judge, provider


def test_locomo_judge_runs_n_times_concurrently() -> None:
    judge, provider = _locomo_judge(['{"label": "CORRECT"}'], runs=5)
    outcome = asyncio.run(judge.judge(_qa(), "gen"))
    assert len(outcome.verdicts) == 5
    assert len(provider.calls) == 5
    # All five should parse as CORRECT.
    assert [v["correct"] for v in outcome.verdicts] == [True] * 5


def test_locomo_judge_mixes_correct_and_wrong_verdicts() -> None:
    judge, _ = _locomo_judge(
        ['{"label": "CORRECT"}', '{"label": "WRONG"}', '{"label": "CORRECT"}'], runs=3
    )
    outcome = asyncio.run(judge.judge(_qa(), "gen"))
    corrects = [v["correct"] for v in outcome.verdicts]
    assert corrects.count(True) == 2
    assert corrects.count(False) == 1
    assert locomo_majority_correct(outcome.verdicts) is True


def test_locomo_judge_prompt_fingerprint_is_constant() -> None:
    """Single template means fp doesn't depend on the qa."""

    judge, _ = _locomo_judge(['{"label": "CORRECT"}'])
    qa1 = _qa(question_id="qa_0")
    qa2 = _qa(question_id="qa_1")
    assert judge.prompt_fingerprint(qa1) == judge.prompt_fingerprint(qa2)
    assert judge.prompt_fingerprint(qa1) == LOCOMO_PROMPT_FINGERPRINTS["locomo"]


def test_locomo_judge_embeds_same_fingerprint_in_outcome() -> None:
    judge, _ = _locomo_judge(['{"label": "CORRECT"}'])
    outcome = asyncio.run(judge.judge(_qa(), "gen"))
    assert outcome.prompt_fingerprint == LOCOMO_PROMPT_FINGERPRINTS["locomo"]


def test_locomo_judge_rejects_zero_runs() -> None:
    provider = _MultiFakeProvider(['{"label": "CORRECT"}'])
    client = JudgeClient(provider, temperature=0.0)
    with pytest.raises(ValueError, match="runs >= 1"):
        LocomoJudge(client, runs=0, temperature=0.0, bundle_fingerprint=LOCOMO_JUDGE_FINGERPRINT)


def test_locomo_judge_reports_time() -> None:
    judge, _ = _locomo_judge(['{"label": "CORRECT"}'])
    outcome = asyncio.run(judge.judge(_qa(), "gen"))
    assert outcome.judge_time_ms >= 0.0


def _beam_judge(response: str = "yes") -> tuple[BeamJudge, _FakeProvider]:
    provider = _FakeProvider(response)
    client = JudgeClient(provider, temperature=0.0)
    judge = BeamJudge(
        client,
        runs=1,
        temperature=0.0,
        bundle_fingerprint=BEAM_JUDGE_FINGERPRINT,
    )
    return judge, provider


def _beam_qa(ability: str, *, question_id: str = "q1") -> QAItem:
    return QAItem(
        question_id=question_id,
        question="Q?",
        gold="A",
        question_type=ability,
    )


def test_beam_judge_parses_yes() -> None:
    judge, _ = _beam_judge("yes")
    outcome = asyncio.run(judge.judge(_beam_qa("multi-hop-reasoning"), "gen"))
    assert outcome.verdicts == [{"correct": True, "raw": "yes"}]


def test_beam_judge_selects_temporal_template() -> None:
    judge, _ = _beam_judge("yes")
    outcome = asyncio.run(judge.judge(_beam_qa("temporal-reasoning"), "19 days"))
    assert outcome.prompt_fingerprint == BEAM_PROMPT_FINGERPRINTS["temporal"]


def test_beam_judge_selects_abstention_template() -> None:
    judge, _ = _beam_judge("yes")
    outcome = asyncio.run(judge.judge(_beam_qa("abstention"), "I don't know"))
    assert outcome.prompt_fingerprint == BEAM_PROMPT_FINGERPRINTS["abstention"]


def test_beam_judge_selects_event_ordering_template() -> None:
    judge, _ = _beam_judge("yes")
    outcome = asyncio.run(judge.judge(_beam_qa("event-ordering"), "A,B,C"))
    assert outcome.prompt_fingerprint == BEAM_PROMPT_FINGERPRINTS["event-ordering"]


def test_beam_judge_falls_back_to_general_for_unknown_ability() -> None:
    judge, _ = _beam_judge("yes")
    outcome = asyncio.run(judge.judge(_beam_qa("mystery-ability"), "gen"))
    assert outcome.prompt_fingerprint == BEAM_PROMPT_FINGERPRINTS["general"]


def test_beam_judge_rejects_multi_run() -> None:
    provider = _FakeProvider("yes")
    client = JudgeClient(provider, temperature=0.0)
    with pytest.raises(ValueError, match="supports --judge-runs 1"):
        BeamJudge(client, runs=3, temperature=0.0, bundle_fingerprint=BEAM_JUDGE_FINGERPRINT)


def test_beam_prompt_fingerprint_matches_judge_write() -> None:
    """Cache-lookup invariant: the fingerprint the orchestrator asks for
    must equal the fingerprint the judge embeds in its outcome."""

    judge, _ = _beam_judge("yes")
    qa = _beam_qa("temporal-reasoning")
    outcome = asyncio.run(judge.judge(qa, "gen"))
    assert judge.prompt_fingerprint(qa) == outcome.prompt_fingerprint


def test_beam_judge_reports_time() -> None:
    judge, _ = _beam_judge("yes")
    outcome = asyncio.run(judge.judge(_beam_qa("knowledge-update"), "gen"))
    assert outcome.judge_time_ms >= 0.0
