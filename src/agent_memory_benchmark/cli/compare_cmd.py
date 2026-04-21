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

Diff logic itself lives in :mod:`agent_memory_benchmark.results.compare`
so the web dashboard can consume the same :class:`CompareTable` rows
without re-implementing scorecard navigation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..results.compare import CompareRow, CompareSection, CompareTable, compare_scorecards


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

    del argv
    a = _load(args.a_path)
    b = _load(args.b_path)
    if a is None or b is None:
        return 1

    table = compare_scorecards(a, b, a_label=str(args.a_path), b_label=str(args.b_path))
    for line in render_compare_text(table):
        print(line)
    return 0


def render_compare_text(table: CompareTable) -> list[str]:
    """Render a :class:`CompareTable` as flat text lines (CLI-friendly)."""

    out: list[str] = [f"# Compare — A={table.a_label}  B={table.b_label}", ""]

    if not table.benchmarks_match:
        out.append(
            f"!! benchmarks differ: A={table.a_benchmark!r}  B={table.b_benchmark!r}"
        )
        out.append("")

    for section in table.sections:
        if not section.non_empty:
            continue
        out.append(f"## {section.title}")
        out.extend(_section_rows(section))
        out.append("")

    return [line for line in out if line is not None]


def _section_rows(section: CompareSection) -> list[str]:
    lines = [_row_header()]
    for row in section.rows:
        lines.append(_format_row(row))
    return lines


def _format_row(row: CompareRow) -> str:
    if row.unit == "pct":
        a_fmt, b_fmt, d_fmt = _fmt_pct(row.a), _fmt_pct(row.b), _fmt_pct_delta(row.delta)
    else:
        a_fmt, b_fmt, d_fmt = _fmt_float(row.a), _fmt_float(row.b), _fmt_float_delta(row.delta)
    return f"| {row.label:<32} | {a_fmt:>10} | {b_fmt:>10} | {d_fmt:>10} |"


def _row_header() -> str:
    return (
        f"| {'metric':<32} | {'A':>10} | {'B':>10} | {'Δ':>10} |\n"
        f"|{'-' * 34}|{'-' * 12}|{'-' * 12}|{'-' * 12}|"
    )


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


__all__ = ["add_compare_subparser", "compare_command", "render_compare_text"]
