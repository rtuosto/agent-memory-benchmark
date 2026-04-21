"""``amb summarize <answers.json>`` — re-render the scorecard from a run.

Reads ``answers.json``, rebuilds a :class:`Scorecard` from the stored
records, and prints a rich-console rendering. Useful when the original
``scorecard.md`` has been lost, when inspecting a paused ``amb run``
whose final artifacts haven't been written yet, or when a user wants
the scorecard printed to a terminal from a CI log.

This command is read-only: it never modifies the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_summarize_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register ``amb summarize`` on the root parser."""

    parser: argparse.ArgumentParser = subparsers.add_parser(
        "summarize",
        help="Rebuild + render the scorecard from an existing answers.json.",
        description=(
            "Re-renders a scorecard without touching the run directory. "
            "Use --format markdown to pipe the output into a PR description; "
            "the default rich console rendering targets terminals."
        ),
    )
    parser.add_argument("answers_path", type=Path, help="Path to an existing answers.json.")
    parser.add_argument(
        "--format",
        choices=("rich", "markdown"),
        default="rich",
        help="Rendering format (default: rich).",
    )
    return parser


def summarize_command(args: argparse.Namespace, *, argv: list[str] | None = None) -> int:
    """Synchronous CLI entry; returns process exit code."""

    del argv  # summarize doesn't care about argv echo
    answers_path: Path = args.answers_path
    if not answers_path.is_file():
        print(f"error: {answers_path} does not exist or is not a file", file=sys.stderr)
        return 2

    # Lazy imports so summarize is cheap to import (no rich dep at parse time).
    from ..results.render import print_scorecard_rich, render_scorecard_markdown
    from ..results.scorecard import build_scorecard
    from ..runner.manifest import load_run_file

    try:
        manifest, record_map = load_run_file(answers_path)
    except (OSError, json.JSONDecodeError, KeyError) as e:
        print(f"error: could not read {answers_path}: {e}", file=sys.stderr)
        return 1

    records = list(record_map.values())
    scorecard = build_scorecard(records, benchmark=manifest.benchmark)

    if args.format == "markdown":
        print(render_scorecard_markdown(scorecard, manifest=manifest))
    else:
        print_scorecard_rich(scorecard, manifest=manifest)
    return 0


__all__ = ["add_summarize_subparser", "summarize_command"]
