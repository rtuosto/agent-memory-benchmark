"""CLI entrypoint. Subcommands are wired in subsequent PRs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amb",
        description="agent-memory-benchmark — measurement instrument for agent memory systems.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    # Subcommands are added in subsequent PRs (run, baseline, rejudge, compare,
    # summarize, cache). The parser is intentionally minimal here so that
    # ``amb --version`` works immediately after the scaffold lands.
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = False
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 0
    # Dispatch will land in PR-8 once the subcommands exist.
    parser.error(f"unknown command: {args.command!r}")
    return 2  # pragma: no cover — parser.error exits


__all__ = ["build_parser", "main"]
