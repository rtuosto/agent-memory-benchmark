"""Tests for ``amb serve`` argument resolution — specifically the
``--tailscale`` bind helper. We stub the CLI shell-out so tests stay
hermetic on machines without tailscale installed.
"""

from __future__ import annotations

import argparse
import subprocess

import pytest

from agent_memory_benchmark.cli import serve_cmd


def _ns(**overrides) -> argparse.Namespace:
    defaults = {"host": None, "tailscale": False}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_bind_is_loopback() -> None:
    bind, url = serve_cmd._resolve_bind(_ns())
    assert bind == "127.0.0.1"
    assert url == "127.0.0.1"


def test_explicit_host_passes_through() -> None:
    bind, url = serve_cmd._resolve_bind(_ns(host="0.0.0.0"))
    assert bind == "0.0.0.0"
    assert url == "0.0.0.0"


def test_tailscale_flag_uses_cli_ip(monkeypatch) -> None:
    monkeypatch.setattr(serve_cmd, "_tailscale_ip", lambda: "100.64.1.5")
    monkeypatch.setattr(
        serve_cmd, "_tailscale_magicdns_name", lambda: "laptop.foo.ts.net"
    )
    bind, url = serve_cmd._resolve_bind(_ns(tailscale=True))
    assert bind == "100.64.1.5"
    assert url == "laptop.foo.ts.net"


def test_tailscale_flag_falls_back_to_ip_when_magicdns_missing(monkeypatch) -> None:
    monkeypatch.setattr(serve_cmd, "_tailscale_ip", lambda: "100.64.1.5")
    monkeypatch.setattr(serve_cmd, "_tailscale_magicdns_name", lambda: None)
    bind, url = serve_cmd._resolve_bind(_ns(tailscale=True))
    assert bind == "100.64.1.5"
    assert url == "100.64.1.5"


def test_tailscale_ip_raises_when_cli_missing(monkeypatch) -> None:
    monkeypatch.setattr(serve_cmd.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="tailscale CLI on PATH"):
        serve_cmd._tailscale_ip()


def test_tailscale_ip_raises_when_daemon_down(monkeypatch) -> None:
    monkeypatch.setattr(serve_cmd.shutil, "which", lambda name: "/usr/bin/tailscale")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=1, stdout="", stderr="daemon not running"
        )

    monkeypatch.setattr(serve_cmd.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="daemon not running"):
        serve_cmd._tailscale_ip()


def test_tailscale_ip_parses_first_ipv4(monkeypatch) -> None:
    monkeypatch.setattr(serve_cmd.shutil, "which", lambda name: "/usr/bin/tailscale")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="100.64.1.5\nfd7a:115c:a1e0::1\n",
            stderr="",
        )

    monkeypatch.setattr(serve_cmd.subprocess, "run", fake_run)
    assert serve_cmd._tailscale_ip() == "100.64.1.5"


def test_tailscale_ip_raises_when_stdout_empty(monkeypatch) -> None:
    monkeypatch.setattr(serve_cmd.shutil, "which", lambda name: "/usr/bin/tailscale")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(serve_cmd.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no IPv4"):
        serve_cmd._tailscale_ip()


def test_magicdns_name_reads_self_dnsname(monkeypatch) -> None:
    monkeypatch.setattr(serve_cmd.shutil, "which", lambda name: "/usr/bin/tailscale")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='{"Self": {"DNSName": "laptop.foo.ts.net."}}',
            stderr="",
        )

    monkeypatch.setattr(serve_cmd.subprocess, "run", fake_run)
    assert serve_cmd._tailscale_magicdns_name() == "laptop.foo.ts.net"


def test_magicdns_name_returns_none_on_missing_cli(monkeypatch) -> None:
    monkeypatch.setattr(serve_cmd.shutil, "which", lambda name: None)
    assert serve_cmd._tailscale_magicdns_name() is None


def test_magicdns_name_returns_none_on_bad_json(monkeypatch) -> None:
    monkeypatch.setattr(serve_cmd.shutil, "which", lambda name: "/usr/bin/tailscale")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="not json", stderr=""
        )

    monkeypatch.setattr(serve_cmd.subprocess, "run", fake_run)
    assert serve_cmd._tailscale_magicdns_name() is None


def test_host_and_tailscale_are_mutually_exclusive() -> None:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="cmd")
    serve_cmd.add_serve_subparser(subs)
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--host", "0.0.0.0", "--tailscale"])
