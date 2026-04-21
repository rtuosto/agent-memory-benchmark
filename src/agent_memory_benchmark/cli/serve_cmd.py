"""``amb serve`` — launch the local web dashboard.

The subcommand is intentionally a thin shell: all real logic lives in
``agent_memory_benchmark.web``. This file only handles argparse wiring,
extras-availability, and the uvicorn invocation. Imports of the web
subpackage stay lazy so ``amb --help`` doesn't crash when ``[web]`` is
not installed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
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
    bind_group = parser.add_mutually_exclusive_group()
    bind_group.add_argument(
        "--host",
        default=None,
        help="Host to bind (default: 127.0.0.1). Use 0.0.0.0 for all interfaces; "
        "prefer --tailscale for mobile access over a tailnet.",
    )
    bind_group.add_argument(
        "--tailscale",
        action="store_true",
        help="Bind to this machine's Tailscale IPv4. Reachable from any other "
        "device on the tailnet (incl. mobile) but not from outside. Requires "
        "the tailscale CLI on PATH and an up daemon.",
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

    try:
        host, url_host = _resolve_bind(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    import uvicorn

    print(
        f"amb serve → http://{url_host}:{args.port}  "
        f"(bind={host}, results={results_dir}, jobs={jobs_dir}, "
        f"max_concurrent={config.max_concurrent_jobs})",
        file=sys.stderr,
    )
    uvicorn.run(
        app,
        host=host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def _resolve_bind(args: argparse.Namespace) -> tuple[str, str]:
    """Return ``(bind_host, url_host)`` for the requested surface.

    ``url_host`` is what we print in the startup banner — for
    ``--tailscale`` we prefer the tailnet MagicDNS name (e.g.
    ``laptop.tailnet.ts.net``) so the user can paste it into a phone.
    Falls back to the IP when MagicDNS isn't available.
    """

    if args.tailscale:
        ip = _tailscale_ip()
        name = _tailscale_magicdns_name() or ip
        return ip, name
    host = args.host or "127.0.0.1"
    return host, host


def _tailscale_ip() -> str:
    """Resolve the local tailnet IPv4 via the ``tailscale`` CLI.

    We intentionally shell out rather than parsing ``/etc/hosts`` or
    pulling in a Tailscale Python client — the CLI is the only
    supported interface on all three platforms we care about
    (Windows/macOS/Linux), and its ``ip -4`` output is stable.
    """

    cli = shutil.which("tailscale")
    if cli is None:
        raise RuntimeError(
            "--tailscale requires the tailscale CLI on PATH. "
            "Install from https://tailscale.com/download and run `tailscale up`."
        )
    try:
        result = subprocess.run(  # noqa: S603  # argv is fixed
            [cli, "ip", "-4"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"tailscale CLI failed: {exc}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "no stderr"
        raise RuntimeError(
            f"`tailscale ip -4` returned {result.returncode}: {stderr}. "
            "Is the daemon running (`tailscale status`)?"
        )
    # Output is one IP per line; take the first IPv4.
    for line in result.stdout.splitlines():
        ip = line.strip()
        if ip and ":" not in ip:  # filter out any stray v6
            return ip
    raise RuntimeError("`tailscale ip -4` returned no IPv4 address.")


def _tailscale_magicdns_name() -> str | None:
    """Best-effort lookup of this node's MagicDNS name.

    Returns ``None`` if ``tailscale status --json`` isn't available or
    doesn't expose a ``Self.DNSName`` — the caller falls back to the
    raw IP, which still works.
    """

    cli = shutil.which("tailscale")
    if cli is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [cli, "status", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        import json

        data = json.loads(result.stdout)
    except (ValueError, json.JSONDecodeError):
        return None
    self_node = data.get("Self") or {}
    dns_name = self_node.get("DNSName")
    if not isinstance(dns_name, str):
        return None
    # DNSName is dot-terminated (RFC style); strip for display.
    return dns_name.rstrip(".") or None


__all__ = ["add_serve_subparser", "serve_command"]
