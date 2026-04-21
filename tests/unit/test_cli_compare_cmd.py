"""Tests for ``cli/compare_cmd.py`` — side-by-side scorecard diff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_memory_benchmark.cli.compare_cmd import (
    _delta,
    _fmt_float_delta,
    _fmt_pct_delta,
    _render_compare,
    compare_command,
)


def _make_scorecard(
    *,
    benchmark: str = "longmemeval",
    overall_accuracy: float | None = 0.7,
    macro_accuracy: float | None = 0.6,
    retrieval_mean: float | None = 42.0,
    per_category: dict[str, dict[str, float | None]] | None = None,
    throughput_qps: float | None = 0.5,
) -> dict[str, Any]:
    per_cat = per_category or {}
    return {
        "benchmark": benchmark,
        "n_questions": 10,
        "n_cases": 2,
        "quality": {
            "overall_accuracy": overall_accuracy,
            "macro_accuracy": macro_accuracy,
            "overall_token_f1": 0.5,
            "per_category": per_cat,
        },
        "latency_ms": {
            "retrieval_per_query": {"mean": retrieval_mean, "p50": retrieval_mean, "p95": retrieval_mean, "max": retrieval_mean, "n": 10} if retrieval_mean is not None else None,
        },
        "retrieval_footprint": {
            "units_per_query": None,
            "tokens_per_query": None,
        },
        "evidence": None,
        "throughput": {
            "queries_per_sec": throughput_qps,
            "sessions_per_sec": None,
        },
    }


class TestDeltaHelpers:
    def test_delta_with_numbers(self) -> None:
        import pytest as _pytest

        assert _delta(0.5, 0.7) == _pytest.approx(0.2)

    def test_delta_with_none_returns_none(self) -> None:
        assert _delta(None, 0.7) is None
        assert _delta(0.5, None) is None

    def test_fmt_pct_delta_positive(self) -> None:
        # 0.02 == 2pp in the output format
        assert _fmt_pct_delta(0.02) == "+2.00pp"

    def test_fmt_pct_delta_negative(self) -> None:
        assert _fmt_pct_delta(-0.015) == "-1.50pp"

    def test_fmt_pct_delta_none(self) -> None:
        assert _fmt_pct_delta(None) == "—"

    def test_fmt_float_delta_positive(self) -> None:
        assert _fmt_float_delta(3.5) == "+3.500"

    def test_fmt_float_delta_negative(self) -> None:
        assert _fmt_float_delta(-0.5) == "-0.500"


class TestRenderCompare:
    def test_produces_quality_rows(self) -> None:
        a = _make_scorecard(overall_accuracy=0.5)
        b = _make_scorecard(overall_accuracy=0.6)
        lines = _render_compare(a, b, a_label="A.json", b_label="B.json")
        text = "\n".join(lines)
        assert "## Quality" in text
        assert "overall_accuracy" in text
        assert "+10.00pp" in text

    def test_flags_benchmark_mismatch(self) -> None:
        a = _make_scorecard(benchmark="longmemeval")
        b = _make_scorecard(benchmark="locomo")
        lines = _render_compare(a, b, a_label="A", b_label="B")
        assert any("benchmarks differ" in ln for ln in lines)

    def test_per_category_union(self) -> None:
        a = _make_scorecard(
            per_category={"cat1": {"accuracy": 0.6, "token_f1": None, "count": 5}}
        )
        b = _make_scorecard(
            per_category={
                "cat1": {"accuracy": 0.7, "token_f1": None, "count": 5},
                "cat2": {"accuracy": 0.5, "token_f1": None, "count": 5},
            }
        )
        lines = _render_compare(a, b, a_label="A", b_label="B")
        text = "\n".join(lines)
        assert "cat1" in text
        assert "cat2" in text  # union shows on both sides
        # cat2 missing on A side should render as —
        cat2_line = next(ln for ln in lines if "cat2" in ln)
        assert "—" in cat2_line

    def test_evidence_section_shows_only_when_present(self) -> None:
        a = _make_scorecard()
        b = _make_scorecard()
        lines = _render_compare(a, b, a_label="A", b_label="B")
        assert not any("Evidence KPIs" in ln for ln in lines)


class TestCompareCommand:
    def test_missing_file_returns_1(self, tmp_path: Path, capsys: Any) -> None:
        import argparse

        args = argparse.Namespace(
            a_path=tmp_path / "missing_a.json",
            b_path=tmp_path / "missing_b.json",
        )
        rc = compare_command(args, argv=None)
        assert rc == 1
        err = capsys.readouterr().err
        assert "does not exist" in err

    def test_happy_path_returns_0(self, tmp_path: Path, capsys: Any) -> None:
        import argparse

        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(_make_scorecard(overall_accuracy=0.5)), encoding="utf-8")
        b_path.write_text(json.dumps(_make_scorecard(overall_accuracy=0.6)), encoding="utf-8")
        rc = compare_command(argparse.Namespace(a_path=a_path, b_path=b_path), argv=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "# Compare" in out
        assert "overall_accuracy" in out

    def test_corrupt_json_returns_1(self, tmp_path: Path, capsys: Any) -> None:
        import argparse

        a_path = tmp_path / "bad.json"
        b_path = tmp_path / "good.json"
        a_path.write_text("{ not json", encoding="utf-8")
        b_path.write_text(json.dumps(_make_scorecard()), encoding="utf-8")
        rc = compare_command(argparse.Namespace(a_path=a_path, b_path=b_path), argv=None)
        assert rc == 1
