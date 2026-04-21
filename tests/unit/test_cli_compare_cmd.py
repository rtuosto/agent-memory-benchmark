"""Tests for ``cli/compare_cmd.py`` — side-by-side scorecard diff.

Low-level diff logic lives in :mod:`agent_memory_benchmark.results.compare`
and has its own suite; here we lock the CLI text shape and the command
entrypoint (exit codes, stderr messages, happy path output).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_memory_benchmark.cli.compare_cmd import (
    _fmt_float_delta,
    _fmt_pct_delta,
    compare_command,
    render_compare_text,
)
from agent_memory_benchmark.results.compare import compare_scorecards


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
    dist = (
        {"mean": retrieval_mean, "p50": retrieval_mean, "p95": retrieval_mean, "max": retrieval_mean, "n": 10}
        if retrieval_mean is not None
        else None
    )
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
        "latency_ms": {"retrieval_per_query": dist},
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
    def test_fmt_pct_delta_positive(self) -> None:
        assert _fmt_pct_delta(0.02) == "+2.00pp"

    def test_fmt_pct_delta_negative(self) -> None:
        assert _fmt_pct_delta(-0.015) == "-1.50pp"

    def test_fmt_pct_delta_none(self) -> None:
        assert _fmt_pct_delta(None) == "—"

    def test_fmt_float_delta_positive(self) -> None:
        assert _fmt_float_delta(3.5) == "+3.500"

    def test_fmt_float_delta_negative(self) -> None:
        assert _fmt_float_delta(-0.5) == "-0.500"


class TestRenderCompareText:
    def test_produces_quality_rows(self) -> None:
        a = _make_scorecard(overall_accuracy=0.5)
        b = _make_scorecard(overall_accuracy=0.6)
        table = compare_scorecards(a, b, a_label="A.json", b_label="B.json")
        text = "\n".join(render_compare_text(table))
        assert "## Quality" in text
        assert "overall_accuracy" in text
        assert "+10.00pp" in text

    def test_flags_benchmark_mismatch(self) -> None:
        a = _make_scorecard(benchmark="longmemeval")
        b = _make_scorecard(benchmark="locomo")
        table = compare_scorecards(a, b, a_label="A", b_label="B")
        text = "\n".join(render_compare_text(table))
        assert "benchmarks differ" in text

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
        table = compare_scorecards(a, b, a_label="A", b_label="B")
        lines = render_compare_text(table)
        text = "\n".join(lines)
        assert "cat1" in text
        assert "cat2" in text
        cat2_line = next(ln for ln in lines if "cat2" in ln)
        assert "—" in cat2_line  # missing on A side renders as —

    def test_evidence_section_hidden_when_empty(self) -> None:
        a = _make_scorecard()
        b = _make_scorecard()
        table = compare_scorecards(a, b, a_label="A", b_label="B")
        lines = render_compare_text(table)
        assert not any("Evidence KPIs" in ln for ln in lines)


class TestCompareCommand:
    def test_missing_file_returns_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        import argparse

        args = argparse.Namespace(
            a_path=tmp_path / "missing_a.json",
            b_path=tmp_path / "missing_b.json",
        )
        rc = compare_command(args, argv=None)
        assert rc == 1
        err = capsys.readouterr().err
        assert "does not exist" in err

    def test_happy_path_returns_0(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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

    def test_corrupt_json_returns_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        import argparse

        a_path = tmp_path / "bad.json"
        b_path = tmp_path / "good.json"
        a_path.write_text("{ not json", encoding="utf-8")
        b_path.write_text(json.dumps(_make_scorecard()), encoding="utf-8")
        rc = compare_command(argparse.Namespace(a_path=a_path, b_path=b_path), argv=None)
        assert rc == 1
