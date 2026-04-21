"""CLI entrypoint. Additional subcommands land in subsequent PRs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..version import __version__
from .run_cmd import add_run_subparser, run_command


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
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = False
    # ``run`` wires in at PR-7; baseline / rejudge / compare / summarize / cache
    # follow in PR-8.
    add_run_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 0
    if args.command == "run":
        return run_command(args, argv=list(argv) if argv is not None else None)
    parser.error(f"unknown command: {args.command!r}")
    return 2  # pragma: no cover — parser.error exits


__all__ = ["build_parser", "main"]
