"""Tests for BEAM judge prompt routing + output parsing."""

from __future__ import annotations

import pytest

from agent_memory_benchmark.judge.beam import (
    BEAM_ABSTENTION_TEMPLATE,
    BEAM_EVENT_ORDERING_TEMPLATE,
    BEAM_GENERAL_TEMPLATE,
    BEAM_PROMPT_FINGERPRINTS,
    BEAM_TEMPORAL_TEMPLATE,
    beam_anscheck_prompt,
    parse_yes_no,
    template_key_for_ability,
)


class TestTemplateKeyRouting:
    def test_temporal_reasoning_routes_to_temporal(self) -> None:
        assert template_key_for_ability("temporal-reasoning") == "temporal"

    def test_event_ordering_routes_to_itself(self) -> None:
        assert template_key_for_ability("event-ordering") == "event-ordering"

    def test_abstention_routes_to_abstention(self) -> None:
        assert template_key_for_ability("abstention") == "abstention"

    @pytest.mark.parametrize(
        "ability",
        [
            "knowledge-update",
            "contradiction-resolution",
            "information-extraction",
            "instruction-following",
            "multi-session-reasoning",
            "preference-following",
            "summarization",
        ],
    )
    def test_generic_abilities_route_to_general(self, ability: str) -> None:
        assert template_key_for_ability(ability) == "general"

    def test_unknown_ability_falls_through_to_general(self) -> None:
        assert template_key_for_ability("mystery-ability") == "general"

    def test_empty_ability_falls_through_to_general(self) -> None:
        assert template_key_for_ability("") == "general"

    def test_ability_is_case_normalized(self) -> None:
        assert template_key_for_ability("Temporal-Reasoning") == "temporal"


class TestPromptFormatting:
    def test_general_prompt_contains_inputs(self) -> None:
        prompt = beam_anscheck_prompt("knowledge-update", "What is 2+2?", "4", "the answer is 4")
        assert "What is 2+2?" in prompt
        assert "the answer is 4" in prompt
        assert prompt.rstrip().endswith("Answer yes or no only.")

    def test_temporal_prompt_mentions_off_by_one(self) -> None:
        prompt = beam_anscheck_prompt("temporal-reasoning", "how many days?", "18", "19 days")
        assert "off-by-one" in prompt

    def test_event_ordering_prompt_mentions_sequence(self) -> None:
        prompt = beam_anscheck_prompt("event-ordering", "order the events", "A,B,C", "B,A,C")
        assert "order" in prompt.lower()

    def test_abstention_prompt_explains_refusal_requirement(self) -> None:
        prompt = beam_anscheck_prompt("abstention", "unanswerable?", "info missing", "I don't know")
        assert "abstain" in prompt.lower()

    def test_general_template_is_used_for_unmapped_ability(self) -> None:
        # If routing drops "knowledge-update" -> general, the formatted
        # prompt should contain a substring unique to BEAM_GENERAL_TEMPLATE.
        prompt = beam_anscheck_prompt("knowledge-update", "q", "a", "r")
        assert "contains the correct answer" in prompt


class TestParseYesNo:
    @pytest.mark.parametrize("text", ["yes", "Yes", "YES.", "yes, correct"])
    def test_yes_variants_return_true(self, text: str) -> None:
        assert parse_yes_no(text) is True

    @pytest.mark.parametrize("text", ["no", "No.", "nope", "the answer is wrong"])
    def test_non_yes_variants_return_false(self, text: str) -> None:
        assert parse_yes_no(text) is False

    def test_whitespace_is_stripped(self) -> None:
        assert parse_yes_no("\n  yes") is True

    def test_yesterday_does_not_count_as_yes(self) -> None:
        """Word-boundary discipline: ``yesterday`` mustn't match ``yes``."""

        assert parse_yes_no("yesterday was fine") is False


class TestFingerprintConsistency:
    def test_fingerprints_match_template_bytes(self) -> None:
        import hashlib

        for key, template in {
            "general": BEAM_GENERAL_TEMPLATE,
            "temporal": BEAM_TEMPORAL_TEMPLATE,
            "event-ordering": BEAM_EVENT_ORDERING_TEMPLATE,
            "abstention": BEAM_ABSTENTION_TEMPLATE,
        }.items():
            expected = hashlib.sha256(template.encode("utf-8")).hexdigest()
            assert BEAM_PROMPT_FINGERPRINTS[key] == expected
