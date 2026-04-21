"""Tests for :mod:`agent_memory_benchmark.cache.m3_guard`."""

from __future__ import annotations

import json
from pathlib import Path

from agent_memory_benchmark.cache.m3_guard import (
    M3GuardMismatch,
    check_answer_cache_versions,
)


def _write_answer(cache_root: Path, key: str, payload: object) -> Path:
    answers_dir = cache_root / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    path = answers_dir / f"{key}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestCheckAnswerCacheVersions:
    def test_empty_cache_returns_no_mismatches(self, tmp_path: Path) -> None:
        assert (
            check_answer_cache_versions(
                tmp_path, memory_system_id="engram", expected_memory_version="v1"
            )
            == []
        )

    def test_all_matching_returns_empty(self, tmp_path: Path) -> None:
        _write_answer(
            tmp_path,
            "k1",
            {"memory_system_id": "engram", "memory_version": "v1", "answer": "x"},
        )
        _write_answer(
            tmp_path,
            "k2",
            {"memory_system_id": "engram", "memory_version": "v1", "answer": "y"},
        )
        assert (
            check_answer_cache_versions(
                tmp_path, memory_system_id="engram", expected_memory_version="v1"
            )
            == []
        )

    def test_detects_version_mismatch(self, tmp_path: Path) -> None:
        _write_answer(
            tmp_path,
            "stale",
            {"memory_system_id": "engram", "memory_version": "v0", "answer": "x"},
        )
        mismatches = check_answer_cache_versions(
            tmp_path, memory_system_id="engram", expected_memory_version="v1"
        )
        assert len(mismatches) == 1
        assert mismatches[0].found == "v0"
        assert mismatches[0].unreadable is False
        assert mismatches[0].path.name == "stale.json"

    def test_skips_entries_from_other_memory_systems(self, tmp_path: Path) -> None:
        _write_answer(
            tmp_path,
            "other",
            {"memory_system_id": "other-mem", "memory_version": "v0", "answer": "x"},
        )
        _write_answer(
            tmp_path,
            "mine",
            {"memory_system_id": "engram", "memory_version": "v1", "answer": "y"},
        )
        assert (
            check_answer_cache_versions(
                tmp_path, memory_system_id="engram", expected_memory_version="v1"
            )
            == []
        )

    def test_unreadable_entry_is_flagged(self, tmp_path: Path) -> None:
        answers_dir = tmp_path / "answers"
        answers_dir.mkdir(parents=True)
        bad = answers_dir / "bad.json"
        bad.write_text("{ not json", encoding="utf-8")
        mismatches = check_answer_cache_versions(
            tmp_path, memory_system_id="engram", expected_memory_version="v1"
        )
        assert mismatches == [M3GuardMismatch(path=bad, found=None, unreadable=True)]

    def test_non_dict_payload_is_flagged(self, tmp_path: Path) -> None:
        _write_answer(tmp_path, "list", [1, 2, 3])
        mismatches = check_answer_cache_versions(
            tmp_path, memory_system_id="engram", expected_memory_version="v1"
        )
        assert len(mismatches) == 1
        assert mismatches[0].unreadable is True

    def test_missing_version_field_counts_as_mismatch(self, tmp_path: Path) -> None:
        _write_answer(tmp_path, "no_version", {"memory_system_id": "engram", "answer": "x"})
        mismatches = check_answer_cache_versions(
            tmp_path, memory_system_id="engram", expected_memory_version="v1"
        )
        assert len(mismatches) == 1
        assert mismatches[0].found is None
        assert mismatches[0].unreadable is False

    def test_returns_sorted_for_stable_reporting(self, tmp_path: Path) -> None:
        for key in ("zz", "aa", "mm"):
            _write_answer(
                tmp_path,
                key,
                {"memory_system_id": "engram", "memory_version": "v0", "answer": ""},
            )
        mismatches = check_answer_cache_versions(
            tmp_path, memory_system_id="engram", expected_memory_version="v1"
        )
        assert [m.path.stem for m in mismatches] == ["aa", "mm", "zz"]
