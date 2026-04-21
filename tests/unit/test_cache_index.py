"""Tests for :mod:`agent_memory_benchmark.cache.index`."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from agent_memory_benchmark.cache.index import (
    CacheIndexWriter,
    clear_all,
    clear_kind,
    gc_older_than,
    index_touch,
    load_index,
)
from agent_memory_benchmark.cache.keys import INDEX_NAME


class TestLoadIndex:
    def test_missing_returns_skeleton(self, tmp_path: Path) -> None:
        data = load_index(tmp_path)
        assert data == {"version": 1, "entries": {}}

    def test_corrupt_file_returns_skeleton(self, tmp_path: Path) -> None:
        (tmp_path / INDEX_NAME).write_text("{ not json", encoding="utf-8")
        data = load_index(tmp_path)
        assert data == {"version": 1, "entries": {}}

    def test_non_dict_payload_returns_skeleton(self, tmp_path: Path) -> None:
        (tmp_path / INDEX_NAME).write_text("[1, 2, 3]", encoding="utf-8")
        data = load_index(tmp_path)
        assert data == {"version": 1, "entries": {}}


class TestIndexTouch:
    def test_inserts_entry(self, tmp_path: Path) -> None:
        index_touch(
            tmp_path,
            kind="answers",
            key="abc",
            path="answers/abc.json",
            meta={"dataset": "longmemeval"},
        )
        data = load_index(tmp_path)
        assert data["entries"]["abc"]["kind"] == "answers"
        assert data["entries"]["abc"]["path"] == "answers/abc.json"
        assert data["entries"]["abc"]["meta"] == {"dataset": "longmemeval"}
        assert data["entries"]["abc"]["updated"].endswith("Z")

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        index_touch(tmp_path, kind="answers", key="abc", path="old.json")
        index_touch(tmp_path, kind="answers", key="abc", path="new.json")
        data = load_index(tmp_path)
        assert data["entries"]["abc"]["path"] == "new.json"


class TestCacheIndexWriter:
    def test_batch_writes_on_flush(self, tmp_path: Path) -> None:
        with CacheIndexWriter(tmp_path) as writer:
            writer.touch(kind="answers", key="a", path="answers/a.json")
            writer.touch(kind="answers", key="b", path="answers/b.json")
            # Nothing written until flush/exit.
            assert not (tmp_path / INDEX_NAME).exists()
        data = load_index(tmp_path)
        assert set(data["entries"]) == {"a", "b"}

    def test_flush_is_idempotent(self, tmp_path: Path) -> None:
        writer = CacheIndexWriter(tmp_path)
        writer.touch(kind="judge", key="j", path="judge/j.json")
        writer.flush()
        writer.flush()  # no-op
        assert set(load_index(tmp_path)["entries"]) == {"j"}

    def test_empty_flush_does_nothing(self, tmp_path: Path) -> None:
        CacheIndexWriter(tmp_path).flush()
        assert not (tmp_path / INDEX_NAME).exists()


class TestClearHelpers:
    def _seed(self, cache_root: Path) -> None:
        (cache_root / "answers").mkdir(parents=True)
        (cache_root / "judge").mkdir(parents=True)
        (cache_root / "ingestion" / "engram" / "k").mkdir(parents=True)
        (cache_root / "answers" / "a.json").write_text("{}", encoding="utf-8")
        (cache_root / "judge" / "j.json").write_text("{}", encoding="utf-8")
        index_touch(cache_root, kind="answers", key="a", path="answers/a.json")
        index_touch(cache_root, kind="judge", key="j", path="judge/j.json")

    def test_clear_all_removes_everything(self, tmp_path: Path) -> None:
        self._seed(tmp_path)
        clear_all(tmp_path)
        assert not (tmp_path / "answers").exists()
        assert not (tmp_path / "judge").exists()
        assert not (tmp_path / "ingestion").exists()
        assert not (tmp_path / INDEX_NAME).exists()

    def test_clear_all_on_missing_root_is_noop(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        clear_all(missing)  # must not raise

    def test_clear_kind_answers(self, tmp_path: Path) -> None:
        self._seed(tmp_path)
        clear_kind(tmp_path, "answers")
        assert not (tmp_path / "answers").exists()
        assert (tmp_path / "judge").exists()
        data = load_index(tmp_path)
        assert "a" not in data["entries"]
        assert "j" in data["entries"]

    def test_clear_kind_unknown_is_noop(self, tmp_path: Path) -> None:
        self._seed(tmp_path)
        clear_kind(tmp_path, "bogus")
        assert (tmp_path / "answers" / "a.json").exists()
        assert set(load_index(tmp_path)["entries"]) == {"a", "j"}


class TestGcOlderThan:
    def _touch_with_age(
        self,
        cache_root: Path,
        *,
        key: str,
        kind: str = "answers",
        age_days: float = 0.0,
        path: str | None = None,
        create_file: bool = True,
    ) -> Path:
        rel = path or f"{kind}/{key}.json"
        abs_path = cache_root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if create_file:
            abs_path.write_text("{}", encoding="utf-8")
        index_touch(cache_root, kind=kind, key=key, path=rel)
        if age_days > 0:
            # Rewrite the timestamp to look aged.
            data = load_index(cache_root)
            aged_epoch = time.time() - age_days * 86400.0
            data["entries"][key]["updated"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(aged_epoch)
            )
            (cache_root / INDEX_NAME).write_text(
                json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
            )
        return abs_path

    def test_removes_old_and_keeps_fresh(self, tmp_path: Path) -> None:
        old = self._touch_with_age(tmp_path, key="old", age_days=60.0)
        fresh = self._touch_with_age(tmp_path, key="fresh", age_days=0.0)
        removed = gc_older_than(tmp_path, max_age_days=30.0)
        assert removed == ["old"]
        assert not old.exists()
        assert fresh.exists()
        data = load_index(tmp_path)
        assert "old" not in data["entries"]
        assert "fresh" in data["entries"]

    def test_handles_missing_file_on_disk(self, tmp_path: Path) -> None:
        self._touch_with_age(tmp_path, key="gone", age_days=60.0, create_file=False)
        removed = gc_older_than(tmp_path, max_age_days=1.0)
        assert removed == ["gone"]

    def test_skips_entry_with_unparseable_timestamp(self, tmp_path: Path) -> None:
        self._touch_with_age(tmp_path, key="k", age_days=60.0)
        data = load_index(tmp_path)
        data["entries"]["k"]["updated"] = "not-a-timestamp"
        (tmp_path / INDEX_NAME).write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        removed = gc_older_than(tmp_path, max_age_days=0.0)
        assert removed == []

    def test_negative_max_age_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            gc_older_than(tmp_path, max_age_days=-1.0)

    def test_accepts_absolute_path_in_entry(self, tmp_path: Path) -> None:
        abs_file = tmp_path / "other" / "x.json"
        abs_file.parent.mkdir(parents=True)
        abs_file.write_text("{}", encoding="utf-8")
        # Record with an absolute path in the entry.
        entries: dict[str, Any] = {
            "x": {
                "kind": "answers",
                "path": str(abs_file),
                "updated": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400 * 60)
                ),
                "meta": {},
            }
        }
        (tmp_path / INDEX_NAME).write_text(
            json.dumps({"version": 1, "entries": entries}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        removed = gc_older_than(tmp_path, max_age_days=30.0)
        assert removed == ["x"]
        assert not abs_file.exists()
