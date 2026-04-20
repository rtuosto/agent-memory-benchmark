"""Smoke tests for the scaffold — prove the package imports and the CLI binds."""

from __future__ import annotations

import subprocess
import sys

import agent_memory_benchmark
from agent_memory_benchmark import __version__
from agent_memory_benchmark.version import HTTP_API_VERSION, PROTOCOL_VERSION


def test_package_exposes_version() -> None:
    assert __version__ == agent_memory_benchmark.__version__
    assert __version__.count(".") == 2


def test_protocol_and_http_versions_defined() -> None:
    assert PROTOCOL_VERSION == "1"
    assert HTTP_API_VERSION == "1"


def test_cli_version_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory_benchmark", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert __version__ in result.stdout
