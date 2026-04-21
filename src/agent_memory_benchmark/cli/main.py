"""CLI entrypoint. All subcommands are registered here and dispatched below."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence

from ..version import __version__
from .baseline_cmd import add_baseline_subparser, baseline_command
from .cache_cmd import add_cache_subparser, cache_command
from .compare_cmd import add_compare_subparser, compare_command
from .rejudge_cmd import add_rejudge_subparser, rejudge_command
from .run_cmd import add_run_subparser, run_command
from .serve_cmd import add_serve_subparser, serve_command
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
    parser.add_argument(
        "--log-level",
        default=os.environ.get("AMB_LOG_LEVEL", "WARNING"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging threshold for benchmark internals (default: WARNING). "
        "Set to INFO to see prompt-truncation warnings inline with the run.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = False
    add_run_subparser(subparsers)
    add_baseline_subparser(subparsers)
    add_rejudge_subparser(subparsers)
    add_compare_subparser(subparsers)
    add_summarize_subparser(subparsers)
    add_cache_subparser(subparsers)
    add_serve_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "log_level", "WARNING"))
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
        "serve": serve_command,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command!r}")
        return 2  # pragma: no cover — parser.error exits
    return handler(args, argv=argv_list)


def _configure_logging(level_name: str) -> None:
    """Attach a stderr handler that shows benchmark-internal warnings.

    Tests (which don't go through ``main``) keep pytest's caplog behavior
    intact because this function only runs from the CLI entrypoint. We
    only attach a handler if none exist yet so repeated ``main(...)``
    calls in the same process don't multiply-log each message.
    """

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level_name)
    # httpx emits per-request INFO lines that flood the console on long runs;
    # clamp it to WARNING unless the user explicitly picked DEBUG.
    if level_name != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)


__all__ = ["build_parser", "main"]
