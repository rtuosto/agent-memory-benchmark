"""Judge prompts + per-benchmark scoring helpers.

Judge prompts are byte-frozen invariants — :mod:`.prompts` exposes the
fingerprint utility and :mod:`.longmemeval` (plus, in later PRs, ``locomo``
and ``beam``) exposes the byte-exact templates used at inference time.

The :class:`~agent_memory_benchmark.llm.judge_client.JudgeClient` from
PR-3 handles the provider-side plumbing (retries, concurrent runs); this
module owns the *text* of the calibration — the thing that must not drift.
"""

from __future__ import annotations

from .longmemeval import (
    LME_ABSTENTION_TEMPLATE,
    LME_GENERAL_TEMPLATE,
    LME_JUDGE_FINGERPRINT,
    LME_KNOWLEDGE_UPDATE_TEMPLATE,
    LME_PREFERENCE_TEMPLATE,
    LME_PROMPT_FINGERPRINTS,
    LME_PROMPT_TEMPLATES,
    LME_TEMPORAL_TEMPLATE,
    LongMemEvalPromptError,
    is_abstention_question,
    longmemeval_anscheck_prompt,
    parse_yes_no,
)
from .prompts import combined_fingerprint, fingerprint

__all__ = [
    "LME_ABSTENTION_TEMPLATE",
    "LME_GENERAL_TEMPLATE",
    "LME_JUDGE_FINGERPRINT",
    "LME_KNOWLEDGE_UPDATE_TEMPLATE",
    "LME_PREFERENCE_TEMPLATE",
    "LME_PROMPT_FINGERPRINTS",
    "LME_PROMPT_TEMPLATES",
    "LME_TEMPORAL_TEMPLATE",
    "LongMemEvalPromptError",
    "combined_fingerprint",
    "fingerprint",
    "is_abstention_question",
    "longmemeval_anscheck_prompt",
    "parse_yes_no",
]
