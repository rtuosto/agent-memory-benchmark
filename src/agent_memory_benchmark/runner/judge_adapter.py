"""Benchmark-specific judge dispatch.

The orchestrator calls one ``BenchmarkJudge.judge(...)`` per question and
gets back a list of :class:`~.manifest.JudgeRun` verdicts plus the
*template* fingerprint (byte-stable lock from ``judge/prompts.py``). The
template fingerprint flows into the judge cache key so re-baselining a
template automatically invalidates its cached verdicts.

Why a Protocol here instead of inline branching in the orchestrator: each
benchmark has its own template-selection + output-parsing logic
(LongMemEval routes five templates by task + abstention, LOCOMO runs one
template N times for majority vote, BEAM specializes per ability across
the ten-way taxonomy). Isolating that behind this interface keeps the
orchestrator benchmark-agnostic.

The :meth:`BenchmarkJudge.prompt_fingerprint` method lets the orchestrator
look up cached verdicts without knowing what template the judge will use
— it just asks the judge what fingerprint would be written for this
``(qa, generated)`` pair and keys the cache off that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ..judge.beam import (
    BEAM_PROMPT_FINGERPRINTS,
    beam_anscheck_prompt,
    template_key_for_ability,
)
from ..judge.beam import (
    parse_yes_no as beam_parse_yes_no,
)
from ..judge.locomo import (
    LOCOMO_PROMPT_FINGERPRINTS,
    locomo_judge_prompt,
    majority_vote,
    parse_locomo_correct,
)
from ..judge.longmemeval import (
    _GENERAL_TASKS,
    LME_PROMPT_FINGERPRINTS,
    is_abstention_question,
    longmemeval_anscheck_prompt,
    parse_yes_no,
)
from ..llm.judge_client import JudgeClient
from ..types import QAItem


@dataclass
class JudgeOutcome:
    """Result of judging one (question, answer) pair."""

    verdicts: list[dict[str, str | bool]]
    """Per-run judge runs — ``[{"correct": bool, "raw": str}, ...]``."""
    prompt_fingerprint: str
    """Fingerprint of the *template* used (not the formatted prompt)."""
    judge_time_ms: float


class BenchmarkJudge(Protocol):
    """Benchmark-specific judge facade used by the orchestrator."""

    @property
    def bundle_fingerprint(self) -> str:
        """Fingerprint of all templates (for the run manifest)."""

    def prompt_fingerprint(self, qa: QAItem) -> str:
        """Return the template fingerprint that :meth:`judge` would use.

        The orchestrator calls this before :meth:`judge` to compute the
        judge cache key and look up any cached verdict; the lookup must
        agree with the value that would be stored after a real judge run.
        """

    async def judge(self, qa: QAItem, generated: str) -> JudgeOutcome:
        """Run one judge pass over ``(qa, generated)``."""


class LongMemEvalJudge:
    """Judge wrapper for LongMemEval's five-template yes/no protocol."""

    benchmark_name = "longmemeval"

    def __init__(
        self,
        client: JudgeClient,
        *,
        runs: int = 1,
        temperature: float = 0.0,
        bundle_fingerprint: str,
    ) -> None:
        if runs != 1:
            # LongMemEval's upstream protocol is single-run yes/no; multi-run
            # majority-vote is a LOCOMO construct. We guard here so the
            # orchestrator's --judge-runs flag never silently changes what
            # LongMemEval means by "judged".
            raise ValueError(
                f"LongMemEvalJudge supports --judge-runs 1; got {runs}. "
                "Use the LOCOMO judge for multi-run majority voting."
            )
        self._client = client
        self._runs = runs
        self._temperature = temperature
        self._bundle_fingerprint = bundle_fingerprint

    @property
    def bundle_fingerprint(self) -> str:
        return self._bundle_fingerprint

    def prompt_fingerprint(self, qa: QAItem) -> str:
        abstention = is_abstention_question(qa.question_id)
        template_key = _template_key_for(qa.question_type, abstention=abstention)
        return LME_PROMPT_FINGERPRINTS[template_key]

    async def judge(self, qa: QAItem, generated: str) -> JudgeOutcome:
        abstention = is_abstention_question(qa.question_id)
        template_key = _template_key_for(qa.question_type, abstention=abstention)
        fp = LME_PROMPT_FINGERPRINTS[template_key]
        prompt = longmemeval_anscheck_prompt(
            qa.question_type,
            qa.question,
            qa.gold,
            generated,
            abstention=abstention,
        )
        t0 = time.perf_counter()
        raw = await self._client.complete(prompt, temperature=self._temperature)
        judge_ms = (time.perf_counter() - t0) * 1000.0
        correct = parse_yes_no(raw)
        return JudgeOutcome(
            verdicts=[{"correct": correct, "raw": raw}],
            prompt_fingerprint=fp,
            judge_time_ms=judge_ms,
        )


class LocomoJudge:
    """Judge wrapper for LOCOMO's ``runs``-way majority-vote protocol.

    LOCOMO's canonical evaluator runs the same CORRECT/WRONG prompt ``N``
    times (default 10) at low temperature and takes the majority. The
    concurrent fanout is handled by
    :meth:`JudgeClient.complete_runs`; parsing folds JSON-label +
    CORRECT/WRONG substring fallbacks (see
    :func:`~..judge.locomo.parse_locomo_correct`). ``majority_vote`` picks
    the overall verdict and is stored as the first element of
    ``verdicts`` — the per-run entries below it retain their individual
    CORRECT/WRONG labels so the scorecard can compute ``judge_std``.

    The single template means ``prompt_fingerprint(qa)`` is constant
    across questions; we still expose the method so the orchestrator's
    cache-key path stays benchmark-agnostic.
    """

    benchmark_name = "locomo"

    def __init__(
        self,
        client: JudgeClient,
        *,
        runs: int = 10,
        temperature: float = 0.0,
        bundle_fingerprint: str,
    ) -> None:
        if runs < 1:
            raise ValueError(f"LocomoJudge requires runs >= 1; got {runs}.")
        self._client = client
        self._runs = runs
        self._temperature = temperature
        self._bundle_fingerprint = bundle_fingerprint
        self._template_fingerprint = LOCOMO_PROMPT_FINGERPRINTS["locomo"]

    @property
    def bundle_fingerprint(self) -> str:
        return self._bundle_fingerprint

    def prompt_fingerprint(self, qa: QAItem) -> str:
        return self._template_fingerprint

    async def judge(self, qa: QAItem, generated: str) -> JudgeOutcome:
        prompt = locomo_judge_prompt(qa.question, qa.gold, generated)
        t0 = time.perf_counter()
        raws = await self._client.complete_runs(
            prompt,
            runs=self._runs,
            temperature=self._temperature,
            json_mode=True,
        )
        judge_ms = (time.perf_counter() - t0) * 1000.0
        verdicts: list[dict[str, str | bool]] = [
            {"correct": parse_locomo_correct(raw), "raw": raw} for raw in raws
        ]
        return JudgeOutcome(
            verdicts=verdicts,
            prompt_fingerprint=self._template_fingerprint,
            judge_time_ms=judge_ms,
        )


class BeamJudge:
    """Judge wrapper for BEAM's ability-routed yes/no protocol.

    BEAM's ten-ability taxonomy maps onto four templates: ``general``
    (default), ``temporal`` (off-by-one tolerance),
    ``event-ordering`` (strict sequence match), and ``abstention``
    (credit only for explicit refusal). Routing uses
    :func:`~..judge.beam.template_key_for_ability` on
    ``qa.question_type`` — abilities outside the canonical ten fall
    through to ``general``, so a dataset with a renamed ability still
    produces a verdict instead of hard-erroring.

    Single-run yes/no like :class:`LongMemEvalJudge`; multi-run
    majority-vote is a LOCOMO-specific construct and guarded out here.
    """

    benchmark_name = "beam"

    def __init__(
        self,
        client: JudgeClient,
        *,
        runs: int = 1,
        temperature: float = 0.0,
        bundle_fingerprint: str,
    ) -> None:
        if runs != 1:
            raise ValueError(
                f"BeamJudge supports --judge-runs 1; got {runs}. "
                "Use the LOCOMO judge for multi-run majority voting."
            )
        self._client = client
        self._runs = runs
        self._temperature = temperature
        self._bundle_fingerprint = bundle_fingerprint

    @property
    def bundle_fingerprint(self) -> str:
        return self._bundle_fingerprint

    def prompt_fingerprint(self, qa: QAItem) -> str:
        key = template_key_for_ability(qa.question_type)
        return BEAM_PROMPT_FINGERPRINTS[key]

    async def judge(self, qa: QAItem, generated: str) -> JudgeOutcome:
        key = template_key_for_ability(qa.question_type)
        fp = BEAM_PROMPT_FINGERPRINTS[key]
        prompt = beam_anscheck_prompt(qa.question_type, qa.question, qa.gold, generated)
        t0 = time.perf_counter()
        raw = await self._client.complete(prompt, temperature=self._temperature)
        judge_ms = (time.perf_counter() - t0) * 1000.0
        correct = beam_parse_yes_no(raw)
        return JudgeOutcome(
            verdicts=[{"correct": correct, "raw": raw}],
            prompt_fingerprint=fp,
            judge_time_ms=judge_ms,
        )


def _template_key_for(task: str, *, abstention: bool) -> str:
    if abstention:
        return "abstention"
    if task in _GENERAL_TASKS:
        return "general"
    if task in ("temporal-reasoning", "knowledge-update", "single-session-preference"):
        return task
    raise ValueError(f"Unsupported LongMemEval question_type: {task!r}")


def locomo_majority_correct(verdicts: list[dict[str, str | bool]]) -> bool:
    """Convenience: apply :func:`majority_vote` to a verdicts list."""

    return majority_vote([bool(v.get("correct", False)) for v in verdicts])


__all__ = [
    "BeamJudge",
    "BenchmarkJudge",
    "JudgeOutcome",
    "LocomoJudge",
    "LongMemEvalJudge",
    "locomo_majority_correct",
]
