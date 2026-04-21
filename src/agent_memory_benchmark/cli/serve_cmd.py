"""``amb serve`` — launch the local web dashboard.

The subcommand is intentionally a thin shell: all real logic lives in
``agent_memory_benchmark.web``. This file only handles argparse wiring,
extras-availability, and the uvicorn invocation. Imports of the web
subpackage stay lazy so ``amb --help`` doesn't crash when ``[web]`` is
not installed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def add_serve_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "serve",
        help="Launch the local web dashboard (requires the [web] extra).",
        description=(
            "Start a FastAPI dashboard for browsing runs, comparing scorecards, "
            "and launching benchmarks. Binds 127.0.0.1 by default — this is a "
            "local tool, no auth."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1). Do not expose publicly — no auth.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000).")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing <timestamp>_<...>/ run dirs. Default: ./results.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("jobs"),
        help="Directory for job logs + metadata. Created if missing. Default: ./jobs.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent benchmark jobs (default: 1).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload. Dev only.",
    )
    return parser


def serve_command(args: argparse.Namespace, *, argv: Sequence[str] | None = None) -> int:
    """Dispatch for ``amb serve``.

    Unlike ``amb run``, this subcommand blocks — uvicorn owns the process
    until Ctrl-C. Returns 1 if the ``[web]`` extras are missing so CI
    catches misconfigured environments without hanging.
    """

    from ..web.app import WebConfig, create_app, web_deps_available

    available, hint = web_deps_available()
    if not available:
        print(hint, file=sys.stderr)
        return 1

    results_dir = args.results_dir.resolve()
    jobs_dir = args.jobs_dir.resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    config = WebConfig(
        results_dir=results_dir,
        jobs_dir=jobs_dir,
        max_concurrent_jobs=max(1, args.max_concurrent),
    )
    app = create_app(config)

    import uvicorn

    print(
        f"amb serve → http://{args.host}:{args.port}  "
        f"(results={results_dir}, jobs={jobs_dir}, max_concurrent={config.max_concurrent_jobs})",
        file=sys.stderr,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


__all__ = ["add_serve_subparser", "serve_command"]
