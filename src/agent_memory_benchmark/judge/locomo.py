"""LOCOMO judge prompt — byte-stable port of the upstream Mem0 template.

One template, invoked N times per question (default 10) for majority-vote
scoring. The template is :func:`str.format`-style with three named
placeholders: ``{question}``, ``{gold_answer}``, ``{generated_answer}``.

Fingerprint locked against ``tests/unit/test_judge_prompts_stable.py``. A
template drift surfaces as a failing test — re-baselining procedure is
documented there.
"""

from __future__ import annotations

import json
import re

from .prompts import combined_fingerprint, fingerprint

LOCOMO_JUDGE_USER_TEMPLATE = """Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given
the following data: (1) a question (posed by one user to another user), (2) a 'gold'
(ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other
user based on their prior conversations. The gold answer will usually be a concise and
short answer that includes the referenced topic, for example:

Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading -
as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The
generated answer might be much longer or use relative time references (like 'last Tuesday'
or 'next month'), but you should be generous with your grading - as long as it refers to the
same date or time period as the gold answer, it should be counted as CORRECT. Even if the
format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

Now it's time for the real question:

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with
CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will
break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""

LOCOMO_PROMPT_TEMPLATES: dict[str, str] = {"locomo": LOCOMO_JUDGE_USER_TEMPLATE}

LOCOMO_PROMPT_FINGERPRINTS: dict[str, str] = {
    key: fingerprint(template) for key, template in LOCOMO_PROMPT_TEMPLATES.items()
}

LOCOMO_JUDGE_FINGERPRINT: str = combined_fingerprint(LOCOMO_PROMPT_TEMPLATES)


def locomo_judge_prompt(question: str, gold_answer: str, generated_answer: str) -> str:
    """Format the LOCOMO judge prompt for one ``(Q, gold, generated)`` triple."""

    return LOCOMO_JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )


_JSON_LABEL_RE = re.compile(r"\{[^{}]*\"label\"[^{}]*\}", re.DOTALL)


def _extract_json_label(text: str) -> str | None:
    """Return the ``label`` string from a JSON blob if we can find one."""

    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "label" in obj:
            return str(obj["label"])
    except json.JSONDecodeError:
        pass
    m = _JSON_LABEL_RE.search(stripped)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "label" in obj:
                return str(obj["label"])
        except json.JSONDecodeError:
            pass
    return None


def parse_locomo_correct(label_text: str) -> bool:
    """Map raw judge output to a binary CORRECT/WRONG verdict.

    Preferred path: parse the JSON ``{"label": "CORRECT" | "WRONG"}`` the
    template instructs the judge to emit. Fallback: scan the raw text for
    ``CORRECT`` / ``WRONG`` tokens. When both tokens appear (prompt
    violation) or neither does, returns ``False`` — a judge that can't
    make up its mind is treated as ``WRONG``.
    """

    parsed = _extract_json_label(label_text)
    blob = (parsed or label_text).upper()
    has_correct = "CORRECT" in blob
    has_wrong = "WRONG" in blob
    if has_correct and not has_wrong:
        return True
    if has_wrong:
        return False
    return False


def majority_vote(verdicts: list[bool]) -> bool:
    """Strict majority of ``True`` across ``verdicts``.

    Ties resolve to ``False`` — a judge run that can't muster a majority of
    CORRECT verdicts is treated as WRONG.
    """

    if not verdicts:
        return False
    return sum(1 for v in verdicts if v) * 2 > len(verdicts)


__all__ = [
    "LOCOMO_JUDGE_FINGERPRINT",
    "LOCOMO_JUDGE_USER_TEMPLATE",
    "LOCOMO_PROMPT_FINGERPRINTS",
    "LOCOMO_PROMPT_TEMPLATES",
    "locomo_judge_prompt",
    "majority_vote",
    "parse_locomo_correct",
]
