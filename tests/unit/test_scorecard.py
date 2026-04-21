"""Scorecard assembly + KPI shape regression tests."""

from __future__ import annotations

from typing import Any

import pytest

from agent_memory_benchmark.results.scorecard import (
    Distribution,
    build_scorecard,
    normalize_answer,
    scorecard_to_dict,
    token_f1,
)
from agent_memory_benchmark.runner.manifest import QARecord


def _record(
    *,
    key: str,
    question_type: str = "single-session-user",
    gold: str = "hello",
    generated: str = "hello world",
    judge_correct: bool | None = True,
    category: int | None = None,
    retrieval_ms: float = 5.0,
    generation_ms: float = 500.0,
    total_ms: float = 510.0,
    discrepancy_ms: float = 5.0,
    ingestion_ms: float = 50.0,
    units: int = 10,
    tokens: int = 1000,
    evidence_turn_ids: list[str] | None = None,
    retrieved_turn_ids: list[str] | None = None,
    replicate_idx: int = 0,
    judge_time_ms: float = 200.0,
) -> QARecord:
    judge_runs: list[dict[str, Any]] = []
    if judge_correct is not None:
        judge_runs.append({"correct": judge_correct, "raw": "yes" if judge_correct else "no"})
    return QARecord(
        key=key,
        benchmark="longmemeval",
        case_id=key.split("::")[0],
        question="Q?",
        gold=gold,
        generated=generated,
        question_id=key.split("::")[0],
        question_type=question_type,
        category=category,
        qa_index=0,
        replicate_idx=replicate_idx,
        ingestion_time_ms=ingestion_ms,
        retrieval_time_ms=retrieval_ms,
        generation_time_ms=generation_ms,
        total_answer_time_ms=total_ms,
        answer_discrepancy_ms=discrepancy_ms,
        units_retrieved=units,
        tokens_retrieved=tokens,
        evidence_turn_ids=evidence_turn_ids or [],
        retrieved_turn_ids=retrieved_turn_ids or [],
        judge_time_ms=judge_time_ms,
        judge_runs=judge_runs,
    )


def test_empty_scorecard_is_all_nones() -> None:
    sc = build_scorecard([], benchmark="longmemeval")
    assert sc.benchmark == "longmemeval"
    assert sc.n_questions == 0
    assert sc.overall_accuracy is None
    assert sc.macro_accuracy is None
    assert sc.per_category == {}


def test_single_correct_record_yields_100pct_accuracy() -> None:
    sc = build_scorecard([_record(key="q1::0")])
    assert sc.overall_accuracy == 1.0
    assert sc.macro_accuracy == 1.0
    assert "single-session-user" in sc.per_category
    cat = sc.per_category["single-session-user"]
    assert cat.count == 1
    assert cat.accuracy == 1.0
    assert cat.token_f1 is not None
    assert 0.0 <= cat.token_f1 <= 1.0


def test_macro_vs_overall_accuracy_diverge_on_size_skew() -> None:
    """Macro should be size-bias-free; overall reflects raw counts."""

    records = [
        _record(key="big::0", question_type="big", judge_correct=False),
        _record(key="big::1", question_type="big", judge_correct=False),
        _record(key="big::2", question_type="big", judge_correct=False),
        _record(key="big::3", question_type="big", judge_correct=True),
        _record(key="small::0", question_type="small", judge_correct=True),
    ]
    sc = build_scorecard(records)
    assert sc.overall_accuracy == pytest.approx(2.0 / 5.0)
    # macro = mean([big=0.25, small=1.00]) = 0.625
    assert sc.macro_accuracy == pytest.approx(0.625)


def test_per_category_uses_category_number_when_present() -> None:
    records = [
        _record(key="cat1::0", category=1, question_type="ignored"),
        _record(key="cat2::0", category=2, question_type="ignored"),
    ]
    sc = build_scorecard(records)
    assert "category_1" in sc.per_category
    assert "category_2" in sc.per_category


def test_token_f1_handles_squad_normalization() -> None:
    # "the book" and "book" should be identical under SQuAD normalization.
    assert token_f1("the book", "book") == pytest.approx(1.0)
    # Punctuation + case must also normalize away.
    assert token_f1("Hello, World!", "hello world") == pytest.approx(1.0)


def test_normalize_answer_strips_articles_punct_case() -> None:
    assert normalize_answer("The Book, Please!") == "book please"


def test_token_f1_on_disjoint_strings_is_zero() -> None:
    assert token_f1("apples", "oranges") == 0.0


def test_ingestion_distribution_only_counts_nonzero() -> None:
    """Cache-hit zeros shouldn't dilute ingestion latency stats."""

    records = [
        _record(key="q1::0", ingestion_ms=0.0),
        _record(key="q2::0", ingestion_ms=100.0),
        _record(key="q3::0", ingestion_ms=200.0),
    ]
    sc = build_scorecard(records)
    assert sc.ingestion_per_session_ms is not None
    assert sc.ingestion_per_session_ms.n == 2


def test_throughput_queries_per_sec_is_reported() -> None:
    records = [_record(key=f"q{i}::0", total_ms=500.0) for i in range(4)]
    sc = build_scorecard(records)
    assert sc.throughput_queries_per_sec == pytest.approx(4 / (4 * 0.5))


def test_evidence_stats_report_zero_when_no_retrieval() -> None:
    """FullContext adapter returns no retrieved units → evidence KPIs skip."""

    records = [
        _record(
            key="q1::0",
            evidence_turn_ids=["sess_1:1"],
            retrieved_turn_ids=[],
        ),
    ]
    sc = build_scorecard(records)
    assert sc.evidence is not None
    assert sc.evidence.n_questions_with_evidence == 1
    assert sc.evidence.n_questions_with_retrieval == 0
    assert sc.evidence.turn_completeness is None


def test_evidence_stats_compute_when_retrieval_present() -> None:
    records = [
        _record(
            key="q1::0",
            evidence_turn_ids=["t1", "t2"],
            retrieved_turn_ids=["t1", "t3"],
        ),
    ]
    sc = build_scorecard(records)
    assert sc.evidence is not None
    assert sc.evidence.turn_completeness is not None
    # completeness = |{t1}| / |{t1, t2}| = 0.5
    assert sc.evidence.turn_completeness.mean == pytest.approx(0.5)
    # density = |{t1}| / |{t1, t3}| = 0.5
    assert sc.evidence.turn_density is not None
    assert sc.evidence.turn_density.mean == pytest.approx(0.5)


def test_replicate_stats_require_multiple_replicate_ids() -> None:
    records_single = [_record(key=f"q{i}::0", replicate_idx=0) for i in range(3)]
    sc = build_scorecard(records_single)
    assert sc.replicate_mean is None
    assert sc.replicate_std is None

    records_multi = [
        _record(key="q1::0::r0", replicate_idx=0, judge_correct=True),
        _record(key="q1::0::r1", replicate_idx=1, judge_correct=False),
    ]
    sc_multi = build_scorecard(records_multi)
    assert sc_multi.replicate_mean is not None
    assert sc_multi.replicate_std is not None


def test_scorecard_to_dict_matches_plan_shape() -> None:
    """Lock the public JSON shape so consumers of scorecard.json don't break."""

    records = [_record(key="q1::0")]
    sc = build_scorecard(records)
    payload = scorecard_to_dict(sc)
    assert payload["benchmark"] == "longmemeval"
    assert "quality" in payload
    assert "latency_ms" in payload
    assert "retrieval_footprint" in payload
    assert "throughput" in payload
    assert "evidence" in payload
    assert set(payload["quality"]) >= {
        "overall_accuracy",
        "macro_accuracy",
        "overall_token_f1",
        "per_category",
    }
    assert set(payload["latency_ms"]) >= {
        "ingestion_per_session",
        "retrieval_per_query",
        "generation_per_query",
        "answer_total_per_query",
        "answer_discrepancy",
        "judge_per_question",
    }
    assert set(payload["throughput"]) == {"queries_per_sec", "sessions_per_sec"}


def test_distribution_to_dict_includes_n() -> None:
    d = Distribution(mean=1.0, p50=1.0, p95=1.0, max=1.0, n=5)
    assert d.as_dict() == {"mean": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0, "n": 5}
