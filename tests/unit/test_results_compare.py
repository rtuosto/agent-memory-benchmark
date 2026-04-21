"""Tests for the shared :func:`compare_scorecards` helper."""

from __future__ import annotations

import pytest

from agent_memory_benchmark.results.compare import compare_scorecards


def _scorecard(**overrides) -> dict:
    base = {
        "benchmark": "longmemeval",
        "n_questions": 10,
        "n_cases": 2,
        "quality": {
            "overall_accuracy": 0.7,
            "macro_accuracy": 0.6,
            "overall_token_f1": 0.5,
            "per_category": {},
        },
        "latency_ms": {
            "ingestion_total": 30000.0,
            "ingestion_per_case": {"mean": 1000.0, "p95": 2000.0},
            "retrieval_per_query": {"mean": 5.0},
            "generation_per_query": {"mean": 500.0},
            "answer_total_per_query": {"mean": 510.0},
            "answer_discrepancy": {"mean": 5.0},
            "judge_per_question": {"mean": 200.0},
        },
        "retrieval_footprint": {
            "units_per_query": {"mean": 10.0},
            "tokens_per_query": {"mean": 200.0},
        },
        "throughput": {"queries_per_sec": 0.5, "sessions_per_sec": 0.1},
        "evidence": None,
    }
    base.update(overrides)
    return base


def test_quality_deltas_are_computed() -> None:
    a = _scorecard(quality={"overall_accuracy": 0.5, "macro_accuracy": 0.4, "overall_token_f1": 0.3, "per_category": {}})
    b = _scorecard(quality={"overall_accuracy": 0.7, "macro_accuracy": 0.6, "overall_token_f1": 0.5, "per_category": {}})
    table = compare_scorecards(a, b)
    overall = next(r for r in table.quality.rows if r.label == "overall_accuracy")
    assert overall.a == 0.5
    assert overall.b == 0.7
    assert overall.delta == pytest.approx(0.2)
    assert overall.unit == "pct"


def test_benchmarks_match_flag() -> None:
    a = _scorecard(benchmark="longmemeval")
    b = _scorecard(benchmark="locomo")
    table = compare_scorecards(a, b)
    assert table.benchmarks_match is False
    assert table.a_benchmark == "longmemeval"
    assert table.b_benchmark == "locomo"


def test_per_category_union_fills_missing_as_none() -> None:
    a = _scorecard(
        quality={
            "overall_accuracy": 0.5,
            "macro_accuracy": 0.4,
            "overall_token_f1": 0.3,
            "per_category": {"cat1": {"accuracy": 0.5}},
        }
    )
    b = _scorecard(
        quality={
            "overall_accuracy": 0.5,
            "macro_accuracy": 0.4,
            "overall_token_f1": 0.3,
            "per_category": {
                "cat1": {"accuracy": 0.7},
                "cat2": {"accuracy": 0.2},
            },
        }
    )
    table = compare_scorecards(a, b)
    labels = [r.label for r in table.per_category.rows]
    assert labels == ["cat1", "cat2"]
    cat2 = table.per_category.rows[1]
    assert cat2.a is None
    assert cat2.b == 0.2


def test_ingestion_section_contains_total_and_per_case_rows() -> None:
    table = compare_scorecards(_scorecard(), _scorecard())
    labels = [r.label for r in table.ingestion.rows]
    assert "ingestion_total_ms" in labels
    assert "ingestion_per_case_mean" in labels
    assert "ingestion_per_case_p95" in labels
    assert "sessions_per_sec" in labels


def test_section_non_empty_property() -> None:
    a = _scorecard(quality={"overall_accuracy": None, "macro_accuracy": None, "overall_token_f1": None, "per_category": {}})
    b = _scorecard(quality={"overall_accuracy": None, "macro_accuracy": None, "overall_token_f1": None, "per_category": {}})
    # With everything None, quality section should be empty-ish.
    table = compare_scorecards(a, b)
    assert not table.quality.non_empty


def test_evidence_section_populates_when_present() -> None:
    payload = {
        "turn_completeness": {"mean": 0.8},
        "turn_density": {"mean": 0.5},
        "unit_completeness": {"mean": 0.6},
        "unit_density": {"mean": 0.4},
        "token_completeness": {"mean": 0.7},
        "token_density": {"mean": 0.3},
    }
    a = _scorecard(evidence=payload)
    b = _scorecard(evidence=payload)
    table = compare_scorecards(a, b)
    assert table.evidence.non_empty
    labels = [r.label for r in table.evidence.rows]
    assert "turn_completeness" in labels
    assert "token_density" in labels


def test_delta_is_none_when_either_side_missing() -> None:
    a = _scorecard(quality={"overall_accuracy": None, "macro_accuracy": 0.5, "overall_token_f1": 0.5, "per_category": {}})
    b = _scorecard()
    table = compare_scorecards(a, b)
    overall = next(r for r in table.quality.rows if r.label == "overall_accuracy")
    assert overall.delta is None
