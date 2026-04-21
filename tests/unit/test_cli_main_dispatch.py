"""Tests for ``cli/main.py`` — argv-to-handler dispatch."""

from __future__ import annotations

from typing import Any

import pytest

from agent_memory_benchmark.cli import main as cli_main


class _Capturer:
    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.calls: list[tuple[str, list[str] | None]] = []

    def __call__(self, args: Any, *, argv: list[str] | None = None) -> int:
        self.calls.append((args.command, argv))
        return self.rc


@pytest.mark.parametrize(
    "argv,expected_command",
    [
        (
            [
                "run",
                "longmemeval",
                "--memory",
                "full-context",
                "--answer-model",
                "ollama:x",
                "--judge-model",
                "ollama:y",
            ],
            "run",
        ),
        (
            [
                "baseline",
                "longmemeval",
                "--answer-model",
                "ollama:x",
                "--judge-model",
                "ollama:y",
            ],
            "baseline",
        ),
        (
            ["rejudge", "foo.json", "--judge-model", "ollama:x"],
            "rejudge",
        ),
        (["compare", "a.json", "b.json"], "compare"),
        (["summarize", "answers.json"], "summarize"),
        (["cache", "info"], "cache"),
    ],
)
def test_main_dispatches(
    argv: list[str],
    expected_command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = _Capturer()
    # Monkey-patch every dispatch target; the expected_command will receive
    # the call, the others must not.
    handlers = ["run_command", "baseline_command", "rejudge_command",
                "compare_command", "summarize_command", "cache_command"]
    for name in handlers:
        monkeypatch.setattr(f"agent_memory_benchmark.cli.main.{name}", cap)

    rc = cli_main.main(argv)
    assert rc == 0
    assert len(cap.calls) == 1
    assert cap.calls[0][0] == expected_command
    assert cap.calls[0][1] == argv


def test_main_prints_help_when_no_command(capsys: Any) -> None:
    rc = cli_main.main([])
    assert rc == 0
    err = capsys.readouterr().err
    assert "usage:" in err.lower()


def test_main_version(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "amb" in out
