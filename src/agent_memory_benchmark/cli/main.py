"""CLI entrypoint. All subcommands are registered here and dispatched below."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..version import __version__
from .baseline_cmd import add_baseline_subparser, baseline_command
from .cache_cmd import add_cache_subparser, cache_command
from .compare_cmd import add_compare_subparser, compare_command
from .rejudge_cmd import add_rejudge_subparser, rejudge_command
from .run_cmd import add_run_subparser, run_command
from .summarize_cmd import add_summarize_subparser, summarize_command


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
    add_run_subparser(subparsers)
    add_baseline_subparser(subparsers)
    add_rejudge_subparser(subparsers)
    add_compare_subparser(subparsers)
    add_summarize_subparser(subparsers)
    add_cache_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 0
    argv_list = list(argv) if argv is not None else None
    dispatch = {
        "run": run_command,
        "baseline": baseline_command,
        "rejudge": rejudge_command,
        "compare": compare_command,
        "summarize": summarize_command,
        "cache": cache_command,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command!r}")
        return 2  # pragma: no cover — parser.error exits
    return handler(args, argv=argv_list)


__all__ = ["build_parser", "main"]
