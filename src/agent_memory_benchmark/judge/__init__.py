"""Judge prompts + per-benchmark scoring helpers.

Judge prompts are byte-frozen invariants — :mod:`.prompts` exposes the
fingerprint utility and :mod:`.longmemeval` / :mod:`.locomo` / :mod:`.beam`
expose the byte-exact templates used at inference time.

The :class:`~agent_memory_benchmark.llm.judge_client.JudgeClient` from
PR-3 handles the provider-side plumbing (retries, concurrent runs); this
module owns the *text* of the calibration — the thing that must not drift.
"""

from __future__ import annotations

from .beam import (
    BEAM_ABSTENTION_TEMPLATE,
    BEAM_EVENT_ORDERING_TEMPLATE,
    BEAM_GENERAL_TEMPLATE,
    BEAM_JUDGE_FINGERPRINT,
    BEAM_PROMPT_FINGERPRINTS,
    BEAM_PROMPT_TEMPLATES,
    BEAM_TEMPORAL_TEMPLATE,
    BeamPromptError,
    beam_anscheck_prompt,
    template_key_for_ability,
)
from .locomo import (
    LOCOMO_JUDGE_FINGERPRINT,
    LOCOMO_JUDGE_USER_TEMPLATE,
    LOCOMO_PROMPT_FINGERPRINTS,
    LOCOMO_PROMPT_TEMPLATES,
    locomo_judge_prompt,
    majority_vote,
    parse_locomo_correct,
)
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
    "BEAM_ABSTENTION_TEMPLATE",
    "BEAM_EVENT_ORDERING_TEMPLATE",
    "BEAM_GENERAL_TEMPLATE",
    "BEAM_JUDGE_FINGERPRINT",
    "BEAM_PROMPT_FINGERPRINTS",
    "BEAM_PROMPT_TEMPLATES",
    "BEAM_TEMPORAL_TEMPLATE",
    "LME_ABSTENTION_TEMPLATE",
    "LME_GENERAL_TEMPLATE",
    "LME_JUDGE_FINGERPRINT",
    "LME_KNOWLEDGE_UPDATE_TEMPLATE",
    "LME_PREFERENCE_TEMPLATE",
    "LME_PROMPT_FINGERPRINTS",
    "LME_PROMPT_TEMPLATES",
    "LME_TEMPORAL_TEMPLATE",
    "LOCOMO_JUDGE_FINGERPRINT",
    "LOCOMO_JUDGE_USER_TEMPLATE",
    "LOCOMO_PROMPT_FINGERPRINTS",
    "LOCOMO_PROMPT_TEMPLATES",
    "BeamPromptError",
    "LongMemEvalPromptError",
    "beam_anscheck_prompt",
    "combined_fingerprint",
    "fingerprint",
    "is_abstention_question",
    "locomo_judge_prompt",
    "longmemeval_anscheck_prompt",
    "majority_vote",
    "parse_locomo_correct",
    "parse_yes_no",
    "template_key_for_ability",
]
