"""Tests for ``cli/cache_cmd.py`` — info / clear / gc subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from agent_memory_benchmark.cache.index import index_touch, load_index
from agent_memory_benchmark.cache.keys import INDEX_NAME
from agent_memory_benchmark.cli.cache_cmd import (
    _parse_duration_days,
    add_cache_subparser,
    cache_command,
)


def _seeded_cache(root: Path) -> None:
    for kind in ("answers", "judge", "ingestion"):
        (root / kind).mkdir(parents=True, exist_ok=True)
    (root / "answers" / "a.json").write_text('{"x": 1}', encoding="utf-8")
    (root / "judge" / "j.json").write_text('{"y": 2}', encoding="utf-8")
    index_touch(root, kind="answers", key="a", path="answers/a.json")
    index_touch(root, kind="judge", key="j", path="judge/j.json")


class TestParseDurationDays:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("7d", 7.0),
            ("7", 7.0),
            ("1.5", 1.5),
            ("12h", 0.5),
            ("30m", 30 / 1440),
            ("  7d  ", 7.0),
            ("7D", 7.0),
        ],
    )
    def test_accepts_supported_forms(self, value: str, expected: float) -> None:
        assert _parse_duration_days(value) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", ["", "xyz", "1w", "-5d", "7dd"])
    def test_rejects_bad_forms(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _parse_duration_days(bad)


class TestCacheInfo:
    def test_missing_root_is_ok(self, tmp_path: Path, capsys: Any) -> None:
        args = argparse.Namespace(
            cache_root=tmp_path / "nope",
            cache_action="info",
        )
        rc = cache_command(args, argv=None)
        assert rc == 0
        assert "does not exist" in capsys.readouterr().out

    def test_reports_entry_counts(self, tmp_path: Path, capsys: Any) -> None:
        _seeded_cache(tmp_path)
        args = argparse.Namespace(cache_root=tmp_path, cache_action="info")
        rc = cache_command(args, argv=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "answers" in out
        assert "judge" in out
        assert "ingestion" in out
        # Exactly one entry each for answers and judge.
        for line in out.splitlines():
            if line.startswith("answers "):
                assert "1" in line
            if line.startswith("judge "):
                assert "1" in line


class TestCacheClear:
    def test_requires_yes(self, tmp_path: Path, capsys: Any) -> None:
        _seeded_cache(tmp_path)
        args = argparse.Namespace(
            cache_root=tmp_path,
            cache_action="clear",
            kind="answers",
            yes=False,
        )
        rc = cache_command(args, argv=None)
        assert rc == 2
        assert "--yes" in capsys.readouterr().err
        assert (tmp_path / "answers" / "a.json").exists()

    def test_clear_kind(self, tmp_path: Path) -> None:
        _seeded_cache(tmp_path)
        args = argparse.Namespace(
            cache_root=tmp_path,
            cache_action="clear",
            kind="answers",
            yes=True,
        )
        rc = cache_command(args, argv=None)
        assert rc == 0
        assert not (tmp_path / "answers").exists()
        assert (tmp_path / "judge").exists()  # untouched

    def test_clear_all(self, tmp_path: Path) -> None:
        _seeded_cache(tmp_path)
        args = argparse.Namespace(
            cache_root=tmp_path,
            cache_action="clear",
            kind="all",
            yes=True,
        )
        rc = cache_command(args, argv=None)
        assert rc == 0
        assert not (tmp_path / "answers").exists()
        assert not (tmp_path / "judge").exists()
        assert not (tmp_path / INDEX_NAME).exists()

    def test_missing_root_is_noop(self, tmp_path: Path, capsys: Any) -> None:
        args = argparse.Namespace(
            cache_root=tmp_path / "nope",
            cache_action="clear",
            kind="all",
            yes=True,
        )
        rc = cache_command(args, argv=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "does not exist" in out


class TestCacheGc:
    def test_rejects_bad_duration(self, tmp_path: Path, capsys: Any) -> None:
        _seeded_cache(tmp_path)
        args = argparse.Namespace(
            cache_root=tmp_path,
            cache_action="gc",
            before="not-a-duration",
        )
        rc = cache_command(args, argv=None)
        assert rc == 2
        assert "could not parse" in capsys.readouterr().err

    def test_gc_removes_old(self, tmp_path: Path, capsys: Any) -> None:
        _seeded_cache(tmp_path)
        # Age the answers entry.
        data = load_index(tmp_path)
        data["entries"]["a"]["updated"] = "2000-01-01T00:00:00Z"
        (tmp_path / INDEX_NAME).write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        args = argparse.Namespace(
            cache_root=tmp_path,
            cache_action="gc",
            before="30d",
        )
        rc = cache_command(args, argv=None)
        assert rc == 0
        assert "removed 1" in capsys.readouterr().out
        assert not (tmp_path / "answers" / "a.json").exists()
        assert (tmp_path / "judge" / "j.json").exists()

    def test_missing_root_is_noop(self, tmp_path: Path, capsys: Any) -> None:
        args = argparse.Namespace(
            cache_root=tmp_path / "nope",
            cache_action="gc",
            before="7d",
        )
        rc = cache_command(args, argv=None)
        assert rc == 0


class TestCacheSubparser:
    def test_info_parses(self) -> None:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        add_cache_subparser(sub)
        args = parser.parse_args(["cache", "info"])
        assert args.cache_action == "info"

    def test_clear_requires_kind(self) -> None:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        add_cache_subparser(sub)
        with pytest.raises(SystemExit):
            parser.parse_args(["cache", "clear"])

    def test_gc_requires_before(self) -> None:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        add_cache_subparser(sub)
        with pytest.raises(SystemExit):
            parser.parse_args(["cache", "gc"])
