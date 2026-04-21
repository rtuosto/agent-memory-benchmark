"""Benchmark-specific judge dispatch.

The orchestrator calls one ``BenchmarkJudge.judge(...)`` per question and
gets back a list of :class:`~.manifest.JudgeRun` verdicts plus the
*template* fingerprint (byte-stable lock from ``judge/prompts.py``). The
template fingerprint flows into the judge cache key so re-baselining a
template automatically invalidates its cached verdicts.

Why a Protocol here instead of inline branching in the orchestrator: the
LOCOMO and BEAM judges land in later PRs and will need their own
template-selection + output-parsing logic (JSON majority vote for LOCOMO,
ability-specific prompts for BEAM). Isolating that behind this interface
keeps the orchestrator benchmark-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

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


def _template_key_for(task: str, *, abstention: bool) -> str:
    if abstention:
        return "abstention"
    if task in _GENERAL_TASKS:
        return "general"
    if task in ("temporal-reasoning", "knowledge-update", "single-session-preference"):
        return task
    raise ValueError(f"Unsupported LongMemEval question_type: {task!r}")


__all__ = [
    "BenchmarkJudge",
    "JudgeOutcome",
    "LongMemEvalJudge",
]
