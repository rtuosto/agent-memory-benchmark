"""``amb compare A.json B.json`` — diff two ``scorecard.json`` files.

Prints a human-readable side-by-side table so regressions and
improvements jump out of PR descriptions. The output is plain text
(markdown-compatible when pasted into a review) — no rich dependency —
because ``amb compare`` is frequently invoked inside CI loops.

Convention: ``A`` is the baseline / previous run, ``B`` is the
candidate. ``Δ = B − A``; a higher number is good for accuracy and
throughput, worse for latency, and context-dependent for footprint (no
sign interpretation is printed). The raw delta is always reported; the
reader decides.

Missing scalars print as ``—``. Missing keys on one side are rendered
with the empty placeholder rather than raising, so you can compare runs
with different per-category breakdowns without a schema migration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def add_compare_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register ``amb compare`` on the root parser."""

    parser: argparse.ArgumentParser = subparsers.add_parser(
        "compare",
        help="Diff two scorecard.json files side-by-side.",
        description=(
            "Compare scorecards from two runs. The first argument is the "
            "baseline (A); the second is the candidate (B). Δ = B − A."
        ),
    )
    parser.add_argument("a_path", type=Path, help="Baseline scorecard.json")
    parser.add_argument("b_path", type=Path, help="Candidate scorecard.json")
    return parser


def compare_command(args: argparse.Namespace, *, argv: list[str] | None = None) -> int:
    """Synchronous CLI entry; returns process exit code."""

    del argv  # compare doesn't care about argv echo
    a = _load(args.a_path)
    b = _load(args.b_path)
    if a is None or b is None:
        return 1

    for line in _render_compare(a, b, a_label=str(args.a_path), b_label=str(args.b_path)):
        print(line)
    return 0


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"error: {path} does not exist or is not a file", file=sys.stderr)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: could not read {path}: {e}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"error: {path} does not contain a scorecard object", file=sys.stderr)
        return None
    return data


def _render_compare(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    a_label: str,
    b_label: str,
) -> list[str]:
    """Return human-readable diff lines for a pair of scorecards."""

    out: list[str] = []
    out.append(f"# Compare — A={a_label}  B={b_label}")
    out.append("")

    a_bench = a.get("benchmark", "?")
    b_bench = b.get("benchmark", "?")
    if a_bench != b_bench:
        out.append(f"!! benchmarks differ: A={a_bench!r}  B={b_bench!r}")
        out.append("")

    out.append("## Quality")
    out.extend(
        _rows_pct(
            [
                ("overall_accuracy", _dig(a, "quality.overall_accuracy"), _dig(b, "quality.overall_accuracy")),
                ("macro_accuracy", _dig(a, "quality.macro_accuracy"), _dig(b, "quality.macro_accuracy")),
                (
                    "overall_token_f1",
                    _dig(a, "quality.overall_token_f1"),
                    _dig(b, "quality.overall_token_f1"),
                ),
            ]
        )
    )
    out.append("")

    per_cat_a = _dig(a, "quality.per_category") or {}
    per_cat_b = _dig(b, "quality.per_category") or {}
    cats = sorted(set(per_cat_a) | set(per_cat_b))
    if cats:
        out.append("## Per-category accuracy")
        rows: list[tuple[str, float | None, float | None]] = []
        for cat in cats:
            rows.append(
                (
                    cat,
                    _safe_get(per_cat_a, cat, "accuracy"),
                    _safe_get(per_cat_b, cat, "accuracy"),
                )
            )
        out.extend(_rows_pct(rows))
        out.append("")

    out.append("## Latency (mean ms)")
    out.extend(
        _rows_float(
            [
                (label, _dig(a, f"latency_ms.{key}.mean"), _dig(b, f"latency_ms.{key}.mean"))
                for label, key in (
                    ("ingestion_per_case", "ingestion_per_case"),
                    ("retrieval_per_query", "retrieval_per_query"),
                    ("generation_per_query", "generation_per_query"),
                    ("answer_total_per_query", "answer_total_per_query"),
                    ("answer_discrepancy", "answer_discrepancy"),
                    ("judge_per_question", "judge_per_question"),
                )
            ]
        )
    )
    out.append("")

    out.append("## Retrieval footprint (mean per query)")
    out.extend(
        _rows_float(
            [
                (
                    "units",
                    _dig(a, "retrieval_footprint.units_per_query.mean"),
                    _dig(b, "retrieval_footprint.units_per_query.mean"),
                ),
                (
                    "tokens",
                    _dig(a, "retrieval_footprint.tokens_per_query.mean"),
                    _dig(b, "retrieval_footprint.tokens_per_query.mean"),
                ),
            ]
        )
    )
    out.append("")

    out.append("## Throughput")
    out.extend(
        _rows_float(
            [
                (
                    "queries_per_sec",
                    _dig(a, "throughput.queries_per_sec"),
                    _dig(b, "throughput.queries_per_sec"),
                ),
                (
                    "sessions_per_sec",
                    _dig(a, "throughput.sessions_per_sec"),
                    _dig(b, "throughput.sessions_per_sec"),
                ),
            ]
        )
    )
    out.append("")

    if _dig(a, "evidence") is not None or _dig(b, "evidence") is not None:
        out.append("## Evidence KPIs (mean)")
        out.extend(
            _rows_pct(
                [
                    (
                        label,
                        _dig(a, f"evidence.{key}.mean"),
                        _dig(b, f"evidence.{key}.mean"),
                    )
                    for label, key in (
                        ("turn_completeness", "turn_completeness"),
                        ("turn_density", "turn_density"),
                        ("unit_completeness", "unit_completeness"),
                        ("unit_density", "unit_density"),
                        ("token_completeness", "token_completeness"),
                        ("token_density", "token_density"),
                    )
                ]
            )
        )
        out.append("")

    return [line for line in out if line is not None]


def _dig(obj: Any, path: str) -> Any:
    """Return ``obj[a][b][c]`` for a dotted ``"a.b.c"`` path; ``None`` on miss."""

    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def _safe_get(d: Any, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def _rows_pct(rows: list[tuple[str, float | None, float | None]]) -> list[str]:
    """Format (label, a, b) triples with values + deltas as percentages."""

    if not rows:
        return []
    lines = [_row_header()]
    for label, a, b in rows:
        delta = _delta(a, b)
        lines.append(
            f"| {label:<32} | {_fmt_pct(a):>10} | {_fmt_pct(b):>10} | {_fmt_pct_delta(delta):>10} |"
        )
    return lines


def _rows_float(rows: list[tuple[str, float | None, float | None]]) -> list[str]:
    """Format (label, a, b) triples with values + raw numeric delta."""

    if not rows:
        return []
    lines = [_row_header()]
    for label, a, b in rows:
        delta = _delta(a, b)
        lines.append(
            f"| {label:<32} | {_fmt_float(a):>10} | {_fmt_float(b):>10} | {_fmt_float_delta(delta):>10} |"
        )
    return lines


def _row_header() -> str:
    return (
        f"| {'metric':<32} | {'A':>10} | {'B':>10} | {'Δ':>10} |\n"
        f"|{'-' * 34}|{'-' * 12}|{'-' * 12}|{'-' * 12}|"
    )


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.2f}%"


def _fmt_pct_delta(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}pp"


def _fmt_float(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _fmt_float_delta(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


__all__ = ["add_compare_subparser", "compare_command"]
