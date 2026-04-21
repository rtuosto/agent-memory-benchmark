"""Tests for :mod:`agent_memory_benchmark.web.charts`."""

from __future__ import annotations

from agent_memory_benchmark.web.charts import build_chart_data


def test_empty_scorecard_returns_empty_buckets() -> None:
    data = build_chart_data({})
    assert data == {
        "per_category": [],
        "latency": [],
        "footprint": [],
        "evidence": [],
    }


def test_per_category_flattens_dict_into_name_rows() -> None:
    scorecard = {
        "quality": {
            "per_category": {
                "single-session-user": {
                    "count": 10,
                    "accuracy": 0.6,
                    "token_f1": 0.5,
                },
                "temporal-reasoning": {
                    "count": 20,
                    "accuracy": 0.3,
                    "token_f1": None,
                },
            }
        }
    }
    data = build_chart_data(scorecard)
    names = [row["name"] for row in data["per_category"]]
    assert "single-session-user" in names
    assert "temporal-reasoning" in names
    # token_f1 nulls survive as None (client skips them).
    temporal = next(r for r in data["per_category"] if r["name"] == "temporal-reasoning")
    assert temporal["token_f1"] is None
    assert temporal["accuracy"] == 0.3


def test_latency_drops_buckets_without_mean() -> None:
    scorecard = {
        "latency_ms": {
            "retrieval_per_query": {"mean": 5.0, "p50": 4.0, "p95": 9.0, "max": 12.0, "n": 30},
            "ingestion_per_case": {"mean": None},
            "generation_per_query": None,
        }
    }
    data = build_chart_data(scorecard)
    names = [row["name"] for row in data["latency"]]
    assert names == ["retrieval_per_query"]


def test_latency_tolerates_pre_rename_ingestion_key() -> None:
    """Old runs (pre-2026-04-21) used ingestion_per_session — keep rendering."""

    scorecard = {
        "latency_ms": {
            "ingestion_per_session": {"mean": 1000.0, "p50": 900.0, "p95": 2000.0, "max": 5000.0}
        }
    }
    data = build_chart_data(scorecard)
    assert [row["name"] for row in data["latency"]] == ["ingestion_per_session"]


def test_footprint_rows_in_fixed_order() -> None:
    scorecard = {
        "retrieval_footprint": {
            "tokens_per_query": {"mean": 100.0, "p50": 90.0, "p95": 200.0, "max": 500.0},
            "units_per_query": {"mean": 5.0, "p50": 5.0, "p95": 10.0, "max": 15.0},
        }
    }
    data = build_chart_data(scorecard)
    # units first, tokens second — regardless of input order.
    assert [row["name"] for row in data["footprint"]] == [
        "units_per_query",
        "tokens_per_query",
    ]


def test_evidence_pairs_completeness_and_density_per_level() -> None:
    scorecard = {
        "evidence": {
            "turn_completeness": {"mean": 0.8},
            "turn_density": {"mean": 0.5},
            "unit_completeness": {"mean": 0.6},
            "unit_density": {"mean": 0.4},
            "token_completeness": {"mean": 0.7},
            "token_density": {"mean": 0.3},
            "n_questions_with_evidence": 30,
        }
    }
    data = build_chart_data(scorecard)
    assert [row["name"] for row in data["evidence"]] == ["turn", "unit", "token"]
    turn_row = data["evidence"][0]
    assert turn_row["completeness"] == 0.8
    assert turn_row["density"] == 0.5


def test_evidence_omits_levels_without_data() -> None:
    scorecard = {
        "evidence": {
            "turn_completeness": {"mean": 0.8},
            "turn_density": {"mean": 0.5},
        }
    }
    data = build_chart_data(scorecard)
    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["name"] == "turn"
