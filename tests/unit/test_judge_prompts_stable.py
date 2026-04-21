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
    BEAM_ABSTENTION_TEMPLATE,
    BEAM_EVENT_ORDERING_TEMPLATE,
    BEAM_GENERAL_TEMPLATE,
    BEAM_JUDGE_FINGERPRINT,
    BEAM_PROMPT_FINGERPRINTS,
    BEAM_PROMPT_TEMPLATES,
    BEAM_TEMPORAL_TEMPLATE,
    LME_ABSTENTION_TEMPLATE,
    LME_GENERAL_TEMPLATE,
    LME_JUDGE_FINGERPRINT,
    LME_KNOWLEDGE_UPDATE_TEMPLATE,
    LME_PREFERENCE_TEMPLATE,
    LME_PROMPT_FINGERPRINTS,
    LME_PROMPT_TEMPLATES,
    LME_TEMPORAL_TEMPLATE,
    LOCOMO_JUDGE_FINGERPRINT,
    LOCOMO_JUDGE_USER_TEMPLATE,
    LOCOMO_PROMPT_FINGERPRINTS,
    LOCOMO_PROMPT_TEMPLATES,
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


_LOCOMO_GOLDEN: dict[str, str] = {
    "locomo": "73ad9d3dc9b755b310cbc77b573afd0086dab47ecc3775f2fb5f72fcc05a5280",
}

_LOCOMO_COMBINED_GOLDEN = "dff1155ec8266d13105fe91348cfdba55fe40c6f0c94600a29532f49ccbb645a"


@pytest.mark.parametrize("name,expected", sorted(_LOCOMO_GOLDEN.items()))
def test_locomo_prompt_fingerprint_is_locked(name: str, expected: str) -> None:
    actual = LOCOMO_PROMPT_FINGERPRINTS[name]
    assert actual == expected, (
        f"LOCOMO judge prompt {name!r} has drifted:\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        "If this change is intentional, re-baseline per the docstring in this file."
    )


def test_locomo_prompt_catalog_is_exactly_one() -> None:
    """LOCOMO uses a single template invoked N times for majority vote."""

    assert set(LOCOMO_PROMPT_TEMPLATES) == set(_LOCOMO_GOLDEN)


def test_locomo_combined_fingerprint_is_locked() -> None:
    assert LOCOMO_JUDGE_FINGERPRINT == _LOCOMO_COMBINED_GOLDEN


def test_locomo_template_has_three_named_placeholders() -> None:
    """LOCOMO template is ``.format``-style with named slots."""

    for key in ("question", "gold_answer", "generated_answer"):
        assert "{" + key + "}" in LOCOMO_JUDGE_USER_TEMPLATE


def test_locomo_template_trailing_instruction_locks_json_label() -> None:
    """The parse path depends on the judge returning ``{"label": ...}``."""

    assert LOCOMO_JUDGE_USER_TEMPLATE.rstrip().endswith(
        'Just return the label CORRECT or WRONG in a json format with the key as "label".'
    )


def test_locomo_and_lme_combined_fingerprints_differ() -> None:
    """Sanity: different template bundles produce different bundle digests."""

    assert LOCOMO_JUDGE_FINGERPRINT != LME_JUDGE_FINGERPRINT


def test_fingerprint_is_pure_sha256_of_utf8_bytes() -> None:
    """Document the invariant that fingerprint() is plain sha256 of UTF-8 bytes."""

    import hashlib

    expected = hashlib.sha256(b"hello").hexdigest()
    assert fingerprint("hello") == expected


_BEAM_GOLDEN: dict[str, str] = {
    "abstention": "00ed31aa00ff120bae248db9e5d1b4f3f44610874f159a0cae246834c10ccd6f",
    "event-ordering": "6fca7bb17398c1562075658cd6ce53ba877ea815c0b2f338d3970139ed5588e7",
    "general": "4d2ba49c03f82b3ca73173a4794e3ca4845ace58e53710f5edbff5096048d010",
    "temporal": "80601d4a1eba2567dfe64838f0421cf126d8de1442fc12dde127d19a4c6f7d2f",
}

_BEAM_COMBINED_GOLDEN = "671c56e0e99d40c92c6dcdb557b688e0323fc98fa7fa43b626ea69fb634c70a4"


@pytest.mark.parametrize("name,expected", sorted(_BEAM_GOLDEN.items()))
def test_beam_prompt_fingerprint_is_locked(name: str, expected: str) -> None:
    actual = BEAM_PROMPT_FINGERPRINTS[name]
    assert actual == expected, (
        f"BEAM judge prompt {name!r} has drifted:\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        "If this change is intentional, re-baseline per the docstring in this file."
    )


def test_beam_prompt_catalog_is_exactly_four() -> None:
    """Guard against sneaking a new template in without a golden digest."""

    assert set(BEAM_PROMPT_TEMPLATES) == set(_BEAM_GOLDEN)


def test_beam_combined_fingerprint_is_locked() -> None:
    assert BEAM_JUDGE_FINGERPRINT == _BEAM_COMBINED_GOLDEN


def test_beam_templates_end_with_yes_or_no() -> None:
    """All four BEAM templates close with an explicit yes/no question."""

    for name, template in BEAM_PROMPT_TEMPLATES.items():
        assert template.rstrip().endswith("Answer yes or no only."), name


def test_beam_templates_have_three_placeholders() -> None:
    for name, template in BEAM_PROMPT_TEMPLATES.items():
        assert template.count("{}") == 3, (
            f"{name!r} template has {template.count('{}')} placeholders, expected 3"
        )


def test_beam_bundle_fingerprint_differs_from_lme_and_locomo() -> None:
    assert BEAM_JUDGE_FINGERPRINT != LME_JUDGE_FINGERPRINT
    assert BEAM_JUDGE_FINGERPRINT != LOCOMO_JUDGE_FINGERPRINT


def test_beam_templates_cover_four_distinct_shapes() -> None:
    """Sanity: the four BEAM templates are byte-distinct — no accidental
    copy-paste that would defeat ability-specific routing."""

    templates = {
        BEAM_GENERAL_TEMPLATE,
        BEAM_TEMPORAL_TEMPLATE,
        BEAM_EVENT_ORDERING_TEMPLATE,
        BEAM_ABSTENTION_TEMPLATE,
    }
    assert len(templates) == 4
