"""LongMemEval judge prompts — byte-stable ports of the upstream templates.

The five templates below are byte-exact ports of ``get_anscheck_prompt`` in
LongMemEval's ``evaluate_qa.py`` (via the predecessor ``~/code/agent-memory``
port). Each template is a :func:`str.format`-style string with three ``{}``
placeholders filled as ``(question, correct_answer_or_rubric, model_response)``.

Fingerprints in :data:`LME_PROMPT_FINGERPRINTS` are locked against the
test at ``tests/unit/test_judge_prompts_stable.py``. A failing test means
one of these templates has drifted even by a single byte — investigate
before re-baselining.

Task-type routing (matches upstream naming, don't rename):

- ``single-session-user`` / ``single-session-assistant`` / ``multi-session``
  → :data:`LME_GENERAL_TEMPLATE`
- ``temporal-reasoning`` → :data:`LME_TEMPORAL_TEMPLATE`
- ``knowledge-update`` → :data:`LME_KNOWLEDGE_UPDATE_TEMPLATE`
- ``single-session-preference`` → :data:`LME_PREFERENCE_TEMPLATE`
- any task with ``_abs`` in the question_id → :data:`LME_ABSTENTION_TEMPLATE`
  (regardless of ``task``)
"""

from __future__ import annotations

import re

from .prompts import combined_fingerprint, fingerprint

LME_GENERAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

LME_TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In addition, "
    "do not penalize off-by-one errors for the number of days. If the question asks for the "
    "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting "
    "19 days when the answer is 18), the model's response is still correct. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

LME_KNOWLEDGE_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required answer.\n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

LME_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
    "answer no. The model does not need to reflect all the points in the rubric. The response "
    "is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
    "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

LME_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. The model "
    "could say that the information is incomplete, or some other information is given but the asked "
    "information is not.\n\n"
    "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)

LME_PROMPT_TEMPLATES: dict[str, str] = {
    "general": LME_GENERAL_TEMPLATE,
    "temporal-reasoning": LME_TEMPORAL_TEMPLATE,
    "knowledge-update": LME_KNOWLEDGE_UPDATE_TEMPLATE,
    "single-session-preference": LME_PREFERENCE_TEMPLATE,
    "abstention": LME_ABSTENTION_TEMPLATE,
}

LME_PROMPT_FINGERPRINTS: dict[str, str] = {
    key: fingerprint(template) for key, template in LME_PROMPT_TEMPLATES.items()
}

LME_JUDGE_FINGERPRINT: str = combined_fingerprint(LME_PROMPT_TEMPLATES)

_GENERAL_TASKS = frozenset({"single-session-user", "single-session-assistant", "multi-session"})


class LongMemEvalPromptError(Exception):
    """Raised when a task/question combination can't be routed to a template."""


def is_abstention_question(question_id: str) -> bool:
    """LongMemEval marks abstention cases with ``_abs`` in the question ID."""

    return "_abs" in question_id


def longmemeval_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool,
) -> str:
    """Build the judge prompt for one LongMemEval question.

    ``abstention`` is a boolean flag rather than being inferred from
    ``task`` because the upstream convention encodes it in the question ID
    (``_abs``) — call :func:`is_abstention_question` on the question ID and
    pass the result in.
    """

    if abstention:
        return LME_ABSTENTION_TEMPLATE.format(question, answer, response)
    template = _select_template(task)
    return template.format(question, answer, response)


def _select_template(task: str) -> str:
    if task in _GENERAL_TASKS:
        return LME_GENERAL_TEMPLATE
    if task == "temporal-reasoning":
        return LME_TEMPORAL_TEMPLATE
    if task == "knowledge-update":
        return LME_KNOWLEDGE_UPDATE_TEMPLATE
    if task == "single-session-preference":
        return LME_PREFERENCE_TEMPLATE
    raise LongMemEvalPromptError(f"Unsupported LongMemEval question_type: {task!r}")


_YES_BOUNDARY_RE = re.compile(r"^\s*yes\b", flags=re.IGNORECASE)


def parse_yes_no(response: str) -> bool:
    """``True`` iff the response starts with the word ``yes``.

    Word boundary: ``"yes"`` matches ``"yes, the model is correct"`` and
    ``"Yes."`` but not ``"yesterday"`` or a mid-sentence ``"eyes"``.
    """

    return _YES_BOUNDARY_RE.match(response.strip()) is not None


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
    "is_abstention_question",
    "longmemeval_anscheck_prompt",
    "parse_yes_no",
]
