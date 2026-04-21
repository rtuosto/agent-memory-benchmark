"""BEAM judge prompts — ability-routed yes/no scoring.

Four templates cover the ten-ability BEAM taxonomy
(:data:`agent_memory_benchmark.datasets.beam.CANONICAL_ABILITIES`):

- :data:`BEAM_GENERAL_TEMPLATE` — default yes/no grader; used by
  ``knowledge-update``, ``contradiction-resolution``,
  ``information-extraction``, ``instruction-following``,
  ``multi-session-reasoning``, ``preference-following``,
  ``summarization``.
- :data:`BEAM_TEMPORAL_TEMPLATE` — grants off-by-one forgiveness on
  day/week/month counts, mirroring LongMemEval's temporal convention.
- :data:`BEAM_EVENT_ORDERING_TEMPLATE` — requires the model to reproduce
  the sequence faithfully; partial-credit ordering is WRONG.
- :data:`BEAM_ABSTENTION_TEMPLATE` — the question is deliberately
  unanswerable; the model gets credit only for explicitly refusing /
  flagging insufficient information.

All templates are :func:`str.format`-style with three positional ``{}``
placeholders filled as ``(question, correct_answer_or_rubric,
model_response)``. Fingerprints in :data:`BEAM_PROMPT_FINGERPRINTS` are
locked against ``tests/unit/test_judge_prompts_stable.py`` — a drift
surfaces as a failing test. Re-baselining requires a bumped
``protocol_version`` and a documented migration (see the docstring of
``test_judge_prompts_stable.py``).

The ten-way taxonomy starts on a generic grader by design; if the user
sees suspect accuracy on a specific ability, add a specialized template
here before tuning the memory system — we want the instrument's
calibration to precede system-under-test tweaks.
"""

from __future__ import annotations

import re

from .prompts import combined_fingerprint, fingerprint

BEAM_GENERAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no.\n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

BEAM_TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In addition, "
    "do not penalize off-by-one errors for the number of days. If the question asks for the "
    "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting "
    "19 days when the answer is 18), the model's response is still correct.\n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

BEAM_EVENT_ORDERING_TEMPLATE = (
    "I will give you a question about the ordering of events, a correct ordered answer, "
    "and a response from a model. The question asks the model to recover the sequence in "
    "which events occurred. Answer yes only if the model's response lists the same events "
    "in the same order as the correct answer. A response that names every event but in a "
    "different order is WRONG; a response that omits events or introduces events not in "
    "the correct answer is WRONG.\n\n"
    "Question: {}\n\nCorrect Ordered Answer: {}\n\nModel Response: {}\n\n"
    "Does the model response reproduce the correct event order? Answer yes or no only."
)

BEAM_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation of why it is unanswerable, "
    "and a response from a model. Please answer yes only if the model explicitly identifies "
    "the question as unanswerable, flags the information as missing, or refuses to answer. "
    "A confident wrong answer is NOT an abstention. A hedged answer that nonetheless commits "
    "to a specific value is NOT an abstention.\n\n"
    "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly abstain? Answer yes or no only."
)

BEAM_PROMPT_TEMPLATES: dict[str, str] = {
    "general": BEAM_GENERAL_TEMPLATE,
    "temporal": BEAM_TEMPORAL_TEMPLATE,
    "event-ordering": BEAM_EVENT_ORDERING_TEMPLATE,
    "abstention": BEAM_ABSTENTION_TEMPLATE,
}

BEAM_PROMPT_FINGERPRINTS: dict[str, str] = {
    key: fingerprint(template) for key, template in BEAM_PROMPT_TEMPLATES.items()
}

BEAM_JUDGE_FINGERPRINT: str = combined_fingerprint(BEAM_PROMPT_TEMPLATES)

# Ability → template key routing. Abilities absent from this map fall
# through to ``"general"`` in :func:`template_key_for_ability`.
_ABILITY_TO_TEMPLATE: dict[str, str] = {
    "temporal-reasoning": "temporal",
    "event-ordering": "event-ordering",
    "abstention": "abstention",
}


class BeamPromptError(Exception):
    """Raised when an ability/question combination can't be routed."""


def template_key_for_ability(ability: str) -> str:
    """Return the template-dict key for a BEAM ability.

    Any ability absent from :data:`_ABILITY_TO_TEMPLATE` routes to
    ``"general"``; the fallback is intentional — the plan's direction
    is to start with a generic grader and specialize only when
    accuracy is suspect.
    """

    normalized = (ability or "").strip().lower()
    return _ABILITY_TO_TEMPLATE.get(normalized, "general")


def beam_anscheck_prompt(ability: str, question: str, answer: str, response: str) -> str:
    """Build the judge prompt for one BEAM question."""

    key = template_key_for_ability(ability)
    template = BEAM_PROMPT_TEMPLATES[key]
    return template.format(question, answer, response)


_YES_BOUNDARY_RE = re.compile(r"^\s*yes\b", flags=re.IGNORECASE)


def parse_yes_no(response: str) -> bool:
    """``True`` iff the response starts with the word ``yes``.

    Mirrors :func:`agent_memory_benchmark.judge.longmemeval.parse_yes_no`;
    duplicated here (not re-exported) so a future BEAM-specific parser
    change can land without touching LongMemEval.
    """

    return _YES_BOUNDARY_RE.match(response.strip()) is not None


__all__ = [
    "BEAM_ABSTENTION_TEMPLATE",
    "BEAM_EVENT_ORDERING_TEMPLATE",
    "BEAM_GENERAL_TEMPLATE",
    "BEAM_JUDGE_FINGERPRINT",
    "BEAM_PROMPT_FINGERPRINTS",
    "BEAM_PROMPT_TEMPLATES",
    "BEAM_TEMPORAL_TEMPLATE",
    "BeamPromptError",
    "beam_anscheck_prompt",
    "parse_yes_no",
    "template_key_for_ability",
]
