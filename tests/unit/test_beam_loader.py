"""Tests for the BEAM loader — real HF schema coverage.

Row-conversion tests exercise :func:`_row_to_case` directly with rows
that match the shape observed on ``Mohammadta/BEAM`` at write time
(conversation-per-row; ``chat`` nested lists; ``probing_questions`` as
a Python-repr string with the ten-way ability taxonomy). Live HF
fetches are stubbed via ``monkeypatch`` on
``agent_memory_benchmark.datasets.beam._load_hf``.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_memory_benchmark.datasets import (
    BEAM_CANONICAL_ABILITIES,
    BEAM_VALID_VARIANTS,
    BeamDataset,
    load_dataset,
)
from agent_memory_benchmark.datasets.beam import (
    CANONICAL_ABILITIES,
    VALID_SPLITS,
    _ability_from_raw,
    _evidence_from_entry,
    _gold_from_entry,
    _parse_chat,
    _parse_probing_questions,
    _row_to_case,
    load_beam,
)
from agent_memory_benchmark.types import BenchmarkCase


def _chat_session(start_id: int, count: int = 4, time_anchor: str | None = None) -> list[dict[str, Any]]:
    """Build a chat session of ``count`` turns starting at the given id.

    First turn carries ``time_anchor``; subsequent turns set it to None
    (matches the real schema where only session-openers have an anchor).
    """

    turns: list[dict[str, Any]] = []
    for i in range(count):
        turns.append(
            {
                "id": start_id + i,
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"turn-{start_id + i}-content",
                "index": f"1,{i + 1}" if i % 2 == 0 else None,
                "question_type": "main_question" if i == 0 else None,
                "time_anchor": time_anchor if i == 0 else None,
            }
        )
    return turns


def _probing_questions(
    *,
    include: list[str] | None = None,
    with_source_ids: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Build a probing_questions bundle covering several abilities."""

    include = include or [
        "temporal_reasoning",
        "event_ordering",
        "abstention",
        "knowledge_update",
        "summarization",
    ]
    bundle: dict[str, list[dict[str, Any]]] = {}
    for ability in include:
        entry: dict[str, Any] = {
            "question": f"sample {ability} question?",
            "difficulty": "easy",
        }
        if ability == "abstention":
            entry["ideal_response"] = "The conversation does not contain that info."
            entry["source_chat_ids"] = None
        elif ability == "summarization":
            entry["ideal_summary"] = "brief summary"
            if with_source_ids:
                entry["source_chat_ids"] = [0, 1, 2]
        elif ability == "knowledge_update":
            entry["answer"] = "updated value"
            if with_source_ids:
                entry["source_chat_ids"] = {"original_info": [5], "updated_info": [8]}
        else:
            entry["answer"] = f"{ability} canonical answer"
            if with_source_ids:
                entry["source_chat_ids"] = [0, 2]
        bundle[ability] = [entry]
    return bundle


def _beam_row(
    *,
    conversation_id: str = "conv_1",
    sessions: list[list[dict[str, Any]]] | None = None,
    probing: dict[str, list[dict[str, Any]]] | str | None = None,
) -> dict[str, Any]:
    sessions = sessions or [
        _chat_session(start_id=0, count=4, time_anchor="March-15-2024"),
        _chat_session(start_id=4, count=4, time_anchor="April-05-2024"),
    ]
    if probing is None:
        probing = _probing_questions()
    # HF delivers probing_questions as a Python-repr string, not JSON —
    # the loader must handle both; use repr() to mimic the tough case.
    probing_field = probing if isinstance(probing, str) else repr(probing)
    return {
        "conversation_id": conversation_id,
        "chat": sessions,
        "probing_questions": probing_field,
    }


class TestParseChat:
    def test_flattens_sessions_and_preserves_ids(self) -> None:
        sessions = _parse_chat(
            [
                _chat_session(start_id=0, count=3, time_anchor="March-15-2024"),
                _chat_session(start_id=3, count=2, time_anchor="April-05-2024"),
            ]
        )
        assert [s.session_index for s in sessions] == [1, 2]
        assert [s.session_time for s in sessions] == ["March-15-2024", "April-05-2024"]
        assert [t.turn_id for t in sessions[0].turns] == ["0", "1", "2"]
        assert [t.turn_id for t in sessions[1].turns] == ["3", "4"]

    def test_drops_turns_missing_required_fields(self) -> None:
        session = [
            {"id": 0, "role": "user", "content": "hello"},
            {"id": 1, "role": "assistant"},  # no content
            {"role": "user", "content": "orphan"},  # no id
        ]
        sessions = _parse_chat([session])
        assert [t.turn_id for t in sessions[0].turns] == ["0"]

    def test_ignores_non_list_top_level_entries(self) -> None:
        assert _parse_chat([None, "not a list"]) == []  # type: ignore[list-item]


class TestProbingQuestionsParser:
    def test_parses_python_repr_string(self) -> None:
        bundle = {"temporal_reasoning": [{"question": "q?", "answer": "a"}]}
        items = _parse_probing_questions(
            repr(bundle), case_id="conv_1", ability_filter=None
        )
        assert len(items) == 1
        assert items[0].question_type == "temporal-reasoning"
        assert items[0].gold == "a"
        assert items[0].question_id == "conv_1:temporal-reasoning:0"

    def test_parses_strict_json_string(self) -> None:
        import json

        bundle = {"event_ordering": [{"question": "q?", "answer": "a"}]}
        items = _parse_probing_questions(
            json.dumps(bundle), case_id="conv_1", ability_filter=None
        )
        assert len(items) == 1
        assert items[0].question_type == "event-ordering"

    def test_accepts_already_parsed_dict(self) -> None:
        bundle = {"abstention": [{"question": "q?", "ideal_response": "idk"}]}
        items = _parse_probing_questions(
            bundle, case_id="conv_1", ability_filter=None
        )
        assert len(items) == 1
        assert items[0].gold == "idk"

    def test_unparseable_string_yields_empty_list(self) -> None:
        items = _parse_probing_questions(
            "{ garbage", case_id="conv_1", ability_filter=None
        )
        assert items == []

    def test_empty_or_non_string_input_yields_empty_list(self) -> None:
        assert _parse_probing_questions(None, case_id="c", ability_filter=None) == []
        assert _parse_probing_questions("", case_id="c", ability_filter=None) == []

    def test_ability_filter_applies(self) -> None:
        bundle = _probing_questions(
            include=["temporal_reasoning", "event_ordering", "abstention"]
        )
        items = _parse_probing_questions(
            bundle,
            case_id="conv_1",
            ability_filter=frozenset({"abstention"}),
        )
        assert [q.question_type for q in items] == ["abstention"]

    def test_unknown_ability_keys_are_silently_skipped(self) -> None:
        bundle = {
            "temporal_reasoning": [{"question": "q?", "answer": "a"}],
            "new_future_ability": [{"question": "q2?", "answer": "a2"}],
        }
        items = _parse_probing_questions(bundle, case_id="conv_1", ability_filter=None)
        assert [q.question_type for q in items] == ["temporal-reasoning"]

    def test_empty_questions_are_dropped(self) -> None:
        bundle = {"temporal_reasoning": [{"question": "", "answer": "a"}]}
        items = _parse_probing_questions(bundle, case_id="conv_1", ability_filter=None)
        assert items == []

    def test_emits_one_qaitem_per_question_per_ability(self) -> None:
        bundle = {
            "temporal_reasoning": [
                {"question": "q1?", "answer": "a1"},
                {"question": "q2?", "answer": "a2"},
            ],
            "abstention": [{"question": "q3?", "ideal_response": "idk"}],
        }
        items = _parse_probing_questions(bundle, case_id="conv_1", ability_filter=None)
        assert len(items) == 3
        abilities = sorted(q.question_type for q in items)
        assert abilities == ["abstention", "temporal-reasoning", "temporal-reasoning"]


class TestGoldAndEvidence:
    def test_gold_prefers_answer(self) -> None:
        entry = {"answer": "A", "ideal_response": "B", "ideal_summary": "C"}
        assert _gold_from_entry(entry) == "A"

    def test_gold_falls_back_to_ideal_response(self) -> None:
        entry = {"ideal_response": "B", "ideal_summary": "C"}
        assert _gold_from_entry(entry) == "B"

    def test_gold_falls_back_to_ideal_summary(self) -> None:
        entry = {"ideal_summary": "C"}
        assert _gold_from_entry(entry) == "C"

    def test_gold_empty_string_when_none_present(self) -> None:
        assert _gold_from_entry({"question": "?"}) == ""

    def test_evidence_from_list_of_ints(self) -> None:
        assert _evidence_from_entry({"source_chat_ids": [3, 5, 8]}) == ("3", "5", "8")

    def test_evidence_from_dict_flattens_values(self) -> None:
        assert _evidence_from_entry(
            {"source_chat_ids": {"original_info": [5], "updated_info": [8, 9]}}
        ) == ("5", "8", "9")

    def test_evidence_none_returns_empty_tuple(self) -> None:
        assert _evidence_from_entry({"source_chat_ids": None}) == ()
        assert _evidence_from_entry({}) == ()


class TestRowToCase:
    def test_builds_case_with_sessions_and_qa(self) -> None:
        case = _row_to_case(_beam_row(), index=0, ability_filter=None)
        assert isinstance(case, BenchmarkCase)
        assert case.case_id == "conv_1"
        assert case.dataset == "beam"
        # Two sessions, 4 turns each.
        assert [len(s.turns) for s in case.sessions] == [4, 4]
        # Five probing questions across five abilities.
        assert len(case.qa) == 5
        assert all(q.question_type in CANONICAL_ABILITIES for q in case.qa)

    def test_turn_ids_are_globally_unique_strings(self) -> None:
        case = _row_to_case(_beam_row(), index=0, ability_filter=None)
        all_ids = [t.turn_id for s in case.sessions for t in s.turns]
        assert all_ids == ["0", "1", "2", "3", "4", "5", "6", "7"]
        assert len(set(all_ids)) == len(all_ids)

    def test_session_time_from_first_turn_anchor(self) -> None:
        case = _row_to_case(_beam_row(), index=0, ability_filter=None)
        assert case.sessions[0].session_time == "March-15-2024"
        assert case.sessions[1].session_time == "April-05-2024"

    def test_ability_filter_restricts_qa_items(self) -> None:
        case = _row_to_case(
            _beam_row(), index=0, ability_filter=frozenset({"abstention"})
        )
        assert [q.question_type for q in case.qa] == ["abstention"]

    def test_evidence_turn_ids_reach_qaitem(self) -> None:
        case = _row_to_case(_beam_row(), index=0, ability_filter=None)
        event_ordering = [q for q in case.qa if q.question_type == "event-ordering"]
        assert event_ordering
        # _probing_questions() sets source_chat_ids=[0, 2] for generic abilities.
        assert event_ordering[0].evidence_turn_ids == ("0", "2")

    def test_knowledge_update_evidence_flattens_original_and_updated(self) -> None:
        case = _row_to_case(_beam_row(), index=0, ability_filter=None)
        ku = [q for q in case.qa if q.question_type == "knowledge-update"]
        assert ku
        assert ku[0].evidence_turn_ids == ("5", "8")

    def test_abstention_has_empty_evidence(self) -> None:
        case = _row_to_case(_beam_row(), index=0, ability_filter=None)
        abs_q = [q for q in case.qa if q.question_type == "abstention"]
        assert abs_q
        assert abs_q[0].evidence_turn_ids == ()

    def test_conversation_id_missing_derives_case_id(self) -> None:
        row = _beam_row()
        row.pop("conversation_id")
        case = _row_to_case(row, index=7, ability_filter=None)
        assert case.case_id == "beam_7"

    def test_ability_names_are_normalized_to_hyphenated_form(self) -> None:
        case = _row_to_case(_beam_row(), index=0, ability_filter=None)
        # All emitted question_types use hyphens, never underscores.
        for q in case.qa:
            assert "_" not in q.question_type

    def test_metadata_carries_ability_and_difficulty(self) -> None:
        case = _row_to_case(_beam_row(), index=0, ability_filter=None)
        q = case.qa[0]
        assert q.metadata["ability"] == q.question_type
        assert q.metadata["difficulty"] == "easy"


class TestBeamDatasetInit:
    def test_rejects_unknown_variant(self) -> None:
        with pytest.raises(ValueError, match="variant must be"):
            BeamDataset([], variant="not-a-variant", revision="r")

    def test_rejects_unknown_abilities(self) -> None:
        with pytest.raises(ValueError, match="Unknown BEAM abilities"):
            BeamDataset([], variant="beam", revision="r", abilities=["totally-made-up"])

    def test_underscore_form_abilities_are_normalized(self) -> None:
        # Users passing temporal_reasoning via CLI should work as if they
        # passed temporal-reasoning — we normalize at the filter boundary.
        ds = BeamDataset(
            [], variant="beam", revision="r", abilities=["temporal_reasoning"]
        )
        assert ds._ability_filter == frozenset({"temporal-reasoning"})  # noqa: SLF001

    def test_accepts_empty_abilities_as_no_filter(self) -> None:
        ds = BeamDataset([_beam_row()], variant="beam", revision="r", abilities=[])
        assert len(ds) == 1

    def test_rejects_negative_limit(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            BeamDataset([], variant="beam", revision="r", limit=-1)

    def test_limit_truncates_row_count(self) -> None:
        rows = [_beam_row(conversation_id=f"c{i}") for i in range(5)]
        ds = BeamDataset(rows, variant="beam", revision="r", limit=3)
        assert len(ds) == 3
        assert [c.case_id for c in ds] == ["c0", "c1", "c2"]

    def test_iteration_order_is_stable(self) -> None:
        rows = [_beam_row(conversation_id=f"c{i}") for i in range(3)]
        ds = BeamDataset(rows, variant="beam", revision="r")
        first = [c.case_id for c in ds]
        second = [c.case_id for c in ds]
        assert first == second == ["c0", "c1", "c2"]

    def test_abilities_filter_propagates_to_emitted_cases(self) -> None:
        rows = [_beam_row(conversation_id="c0")]
        ds = BeamDataset(
            rows, variant="beam", revision="r", abilities=["temporal-reasoning"]
        )
        cases = list(ds)
        assert len(cases) == 1
        assert [q.question_type for q in cases[0].qa] == ["temporal-reasoning"]


class TestDescriptorHash:
    def test_identical_inputs_produce_identical_hash(self) -> None:
        rows = [_beam_row()]
        a = BeamDataset(rows, variant="beam", revision="r")
        b = BeamDataset(rows, variant="beam", revision="r")
        assert a.descriptor_hash() == b.descriptor_hash()

    def test_variant_flips_hash(self) -> None:
        rows = [_beam_row()]
        a = BeamDataset(rows, variant="beam", revision="r")
        b = BeamDataset(rows, variant="beam-10m", revision="r")
        assert a.descriptor_hash() != b.descriptor_hash()

    def test_revision_flips_hash(self) -> None:
        rows = [_beam_row()]
        a = BeamDataset(rows, variant="beam", revision="rev-a")
        b = BeamDataset(rows, variant="beam", revision="rev-b")
        assert a.descriptor_hash() != b.descriptor_hash()

    def test_abilities_filter_flips_hash(self) -> None:
        rows = [_beam_row()]
        a = BeamDataset(rows, variant="beam", revision="r")
        b = BeamDataset(
            rows, variant="beam", revision="r", abilities=["temporal-reasoning"]
        )
        assert a.descriptor_hash() != b.descriptor_hash()

    def test_limit_flips_hash(self) -> None:
        rows = [_beam_row(conversation_id=f"c{i}") for i in range(3)]
        a = BeamDataset(rows, variant="beam", revision="r")
        b = BeamDataset(rows, variant="beam", revision="r", limit=2)
        assert a.descriptor_hash() != b.descriptor_hash()

    def test_split_flips_hash(self) -> None:
        rows = [_beam_row()]
        a = BeamDataset(rows, variant="beam", revision="r", split="100K")
        b = BeamDataset(rows, variant="beam", revision="r", split="1M")
        assert a.descriptor_hash() != b.descriptor_hash()


class TestLoad:
    def test_load_beam_rejects_bad_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_memory_benchmark.datasets.beam as beam_mod

        monkeypatch.setattr(beam_mod, "_load_hf", lambda **kw: [])
        with pytest.raises(ValueError, match="only supports splits"):
            load_beam(variant="beam", split="train")

    def test_load_beam_defaults_to_largest_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_memory_benchmark.datasets.beam as beam_mod

        seen: dict[str, str] = {}

        def fake(**kwargs: Any) -> list[dict[str, Any]]:
            seen.update({k: str(v) for k, v in kwargs.items()})
            return []

        monkeypatch.setattr(beam_mod, "_load_hf", fake)
        load_beam(variant="beam", revision="dummy")
        assert seen["split"] == "1M"

    def test_load_beam_forwards_abilities_and_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_memory_benchmark.datasets.beam as beam_mod

        rows = [_beam_row(conversation_id=f"c{i}") for i in range(3)]
        monkeypatch.setattr(beam_mod, "_load_hf", lambda **kw: rows)
        ds = load_beam(
            variant="beam", revision="dummy", abilities=["abstention"], limit=2
        )
        assert len(ds) == 2  # row-limit (not question-limit)
        for case in ds:
            assert [q.question_type for q in case.qa] == ["abstention"]


class TestPackageSurface:
    def test_canonical_abilities_exposed(self) -> None:
        assert BEAM_CANONICAL_ABILITIES == CANONICAL_ABILITIES
        assert len(BEAM_CANONICAL_ABILITIES) == 10

    def test_canonical_abilities_match_hf_taxonomy(self) -> None:
        """The ten abilities observed on Mohammadta/BEAM at write time.

        This assertion pins the taxonomy so a CANONICAL_ABILITIES edit
        that drifts from the real dataset fails loud."""

        expected = {
            "abstention",
            "contradiction-resolution",
            "event-ordering",
            "information-extraction",
            "instruction-following",
            "knowledge-update",
            "multi-session-reasoning",
            "preference-following",
            "summarization",
            "temporal-reasoning",
        }
        assert set(CANONICAL_ABILITIES) == expected

    def test_valid_variants_exposed(self) -> None:
        assert BEAM_VALID_VARIANTS == ("beam", "beam-10m")

    def test_valid_splits_cover_context_tiers(self) -> None:
        assert VALID_SPLITS["beam"] == ("100K", "500K", "1M")
        assert VALID_SPLITS["beam-10m"] == ("1M", "5M", "10M")


def test_ability_from_raw_normalizes_underscores_and_case() -> None:
    assert _ability_from_raw("temporal_reasoning") == "temporal-reasoning"
    assert _ability_from_raw("TEMPORAL_REASONING") == "temporal-reasoning"
    assert _ability_from_raw("  abstention  ") == "abstention"


def test_load_dataset_beam_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end dispatch through the top-level ``load_dataset``."""

    import agent_memory_benchmark.datasets.beam as beam_mod

    monkeypatch.setattr(beam_mod, "_load_hf", lambda **kw: [])
    ds = load_dataset("beam", variant="beam", revision="dummy")
    assert ds.name == "beam"
