"""Tests for :mod:`agent_memory_benchmark.web.charts`."""

from __future__ import annotations

from agent_memory_benchmark.web.charts import build_chart_data


def test_empty_scorecard_returns_empty_buckets() -> None:
    data = build_chart_data({})
    assert data["per_category"] == []
    assert data["latency"] == []
    assert data["footprint"] == []
    assert data["evidence"] == []
    assert data["has_baseline"] is False


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
    temporal = next(r for r in data["per_category"] if r["name"] == "temporal-reasoning")
    assert temporal["token_f1"] is None
    assert temporal["accuracy"] == 0.3
    # No baseline supplied — baseline_* fields are None.
    assert temporal["baseline_accuracy"] is None


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
        }
    }
    data = build_chart_data(scorecard)
    assert [row["name"] for row in data["evidence"]] == ["turn", "unit", "token"]
    turn_row = data["evidence"][0]
    assert turn_row["completeness"] == 0.8
    assert turn_row["density"] == 0.5


def test_baseline_adds_parallel_fields_to_per_category() -> None:
    current = {
        "quality": {
            "per_category": {
                "cat1": {"accuracy": 0.7, "token_f1": 0.6, "count": 10},
            }
        }
    }
    baseline = {
        "quality": {
            "per_category": {
                "cat1": {"accuracy": 0.5, "token_f1": 0.4, "count": 10},
            }
        }
    }
    data = build_chart_data(current, baseline)
    assert data["has_baseline"] is True
    row = data["per_category"][0]
    assert row["accuracy"] == 0.7
    assert row["baseline_accuracy"] == 0.5
    assert row["baseline_token_f1"] == 0.4


def test_baseline_per_category_unions_buckets() -> None:
    current = {
        "quality": {
            "per_category": {"cat1": {"accuracy": 0.7, "token_f1": 0.6, "count": 10}}
        }
    }
    baseline = {
        "quality": {
            "per_category": {
                "cat1": {"accuracy": 0.5, "token_f1": 0.4, "count": 10},
                "cat2": {"accuracy": 0.9, "token_f1": 0.8, "count": 10},
            }
        }
    }
    data = build_chart_data(current, baseline)
    names = [r["name"] for r in data["per_category"]]
    assert "cat1" in names and "cat2" in names
    cat2 = next(r for r in data["per_category"] if r["name"] == "cat2")
    assert cat2["accuracy"] is None
    assert cat2["baseline_accuracy"] == 0.9


def test_baseline_latency_carries_both_means() -> None:
    current = {
        "latency_ms": {
            "retrieval_per_query": {"mean": 5.0, "p50": 4.0, "p95": 9.0, "max": 12.0}
        }
    }
    baseline = {
        "latency_ms": {
            "retrieval_per_query": {"mean": 12.0, "p50": 10.0, "p95": 20.0, "max": 30.0}
        }
    }
    data = build_chart_data(current, baseline)
    row = data["latency"][0]
    assert row["mean"] == 5.0
    assert row["baseline_mean"] == 12.0
    assert row["baseline_p95"] == 20.0


def test_baseline_latency_includes_bucket_only_on_baseline() -> None:
    current = {"latency_ms": {}}
    baseline = {"latency_ms": {"judge_per_question": {"mean": 200.0}}}
    data = build_chart_data(current, baseline)
    assert len(data["latency"]) == 1
    assert data["latency"][0]["name"] == "judge_per_question"
    assert data["latency"][0]["mean"] is None
    assert data["latency"][0]["baseline_mean"] == 200.0


def test_baseline_evidence_adds_baseline_fields() -> None:
    current = {
        "evidence": {
            "turn_completeness": {"mean": 0.8},
            "turn_density": {"mean": 0.5},
        }
    }
    baseline = {
        "evidence": {
            "turn_completeness": {"mean": 0.6},
            "turn_density": {"mean": 0.3},
        }
    }
    data = build_chart_data(current, baseline)
    row = data["evidence"][0]
    assert row["completeness"] == 0.8
    assert row["baseline_completeness"] == 0.6
    assert row["baseline_density"] == 0.3
