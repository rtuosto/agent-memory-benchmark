"""Byte-stable fingerprint lock for judge prompts.

If one of these assertions fails, a judge prompt template has drifted. That
is a P8 invariant — the benchmark's calibration MUST NOT change silently.
Re-baselining procedure:

1. Confirm the change is intentional (commit message / plan doc).
2. Recompute the fingerprint(s) and update the golden(s) in this file.
3. Bump ``protocol_version`` and note the migration in ``docs/methodology.md``.
4. Any cached judge entries from prior versions are automatically invalidated
   because the prompt fingerprint flows into the judge cache key (PR-4).

The golden digests below were cross-verified against the predecessor
``~/code/agent-memory/benchmark/judge.py`` source at PR-6 commit time.
"""

from __future__ import annotations

import pytest

from agent_memory_benchmark.judge import (
    LME_ABSTENTION_TEMPLATE,
    LME_GENERAL_TEMPLATE,
    LME_JUDGE_FINGERPRINT,
    LME_KNOWLEDGE_UPDATE_TEMPLATE,
    LME_PREFERENCE_TEMPLATE,
    LME_PROMPT_FINGERPRINTS,
    LME_PROMPT_TEMPLATES,
    LME_TEMPORAL_TEMPLATE,
    combined_fingerprint,
    fingerprint,
)

_LONGMEMEVAL_GOLDEN: dict[str, str] = {
    "abstention": "5c0b365a1e1d06db36377c735432b56e122ca3c428f89faf61d43a0d5a7e050b",
    "general": "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc",
    "knowledge-update": "183a9b3a6197ec620940f610cdc1207201ec98c1113dd633ea685cfc322fafac",
    "single-session-preference": "741ee3bcbea7ff5e8ed359acef61d2f8ded3de021bbcff6ee13de455f2e2aa9b",
    "temporal-reasoning": "8d33a5fdd83afeeb4592454a965eab43d1fcb2dedc042d1d3892f4254be6c273",
}

_LONGMEMEVAL_COMBINED_GOLDEN = "33013a6ed6390a0d3aaf520ab1c1fda47c345241b34a47a007ae2362d2eb5628"


@pytest.mark.parametrize("name,expected", sorted(_LONGMEMEVAL_GOLDEN.items()))
def test_longmemeval_prompt_fingerprint_is_locked(name: str, expected: str) -> None:
    actual = LME_PROMPT_FINGERPRINTS[name]
    assert actual == expected, (
        f"LongMemEval judge prompt {name!r} has drifted:\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        "If this change is intentional, re-baseline per the docstring in this file."
    )


def test_longmemeval_prompt_catalog_is_exactly_five() -> None:
    """Guard against sneaking a new template in without a golden digest."""

    assert set(LME_PROMPT_TEMPLATES) == set(_LONGMEMEVAL_GOLDEN)


def test_longmemeval_templates_end_with_placeholder_sentence() -> None:
    """All five templates close with an explicit yes/no question — the judge
    behavior depends on that last-sentence pattern."""

    for template in LME_PROMPT_TEMPLATES.values():
        assert template.rstrip().endswith("Answer yes or no only.")


def test_longmemeval_templates_have_three_placeholders() -> None:
    """All five templates use exactly three ``{}`` substitution slots."""

    for name, template in LME_PROMPT_TEMPLATES.items():
        assert template.count("{}") == 3, (
            f"{name!r} template has {template.count('{}')} placeholders, expected 3"
        )


def test_longmemeval_combined_fingerprint_is_locked() -> None:
    assert LME_JUDGE_FINGERPRINT == _LONGMEMEVAL_COMBINED_GOLDEN


def test_combined_fingerprint_is_order_independent() -> None:
    """Re-ordering the template dict must not change its digest."""

    shuffled = {
        "temporal-reasoning": LME_TEMPORAL_TEMPLATE,
        "abstention": LME_ABSTENTION_TEMPLATE,
        "general": LME_GENERAL_TEMPLATE,
        "knowledge-update": LME_KNOWLEDGE_UPDATE_TEMPLATE,
        "single-session-preference": LME_PREFERENCE_TEMPLATE,
    }
    assert combined_fingerprint(shuffled) == LME_JUDGE_FINGERPRINT


def test_fingerprint_is_pure_sha256_of_utf8_bytes() -> None:
    """Document the invariant that fingerprint() is plain sha256 of UTF-8 bytes."""

    import hashlib

    expected = hashlib.sha256(b"hello").hexdigest()
    assert fingerprint("hello") == expected
