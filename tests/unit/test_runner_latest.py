"""``results/latest`` pointer behavior across platforms.

On POSIX the symlink path should always succeed. On Windows where the
developer doesn't have SeCreateSymbolicLinkPrivilege, we expect the
junction fallback to kick in, and if that also fails the ``latest.txt``
file records the absolute target — never a hard failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent_memory_benchmark.runner.latest import (
    resolve_latest_pointer,
    update_latest_pointer,
)


def _make_target(base: Path, name: str = "run_abc") -> Path:
    target = base / name
    target.mkdir()
    (target / "marker").write_text("hi", encoding="utf-8")
    return target


def test_update_latest_pointer_creates_pointer(tmp_path: Path) -> None:
    target = _make_target(tmp_path)
    pointer = update_latest_pointer(tmp_path, target)
    assert pointer.exists()
    # Pointer should resolve back to the target directory.
    resolved = resolve_latest_pointer(tmp_path)
    assert resolved is not None
    assert resolved.resolve() == target.resolve()


def test_update_latest_pointer_replaces_previous(tmp_path: Path) -> None:
    first = _make_target(tmp_path, "run_1")
    second = _make_target(tmp_path, "run_2")
    update_latest_pointer(tmp_path, first)
    update_latest_pointer(tmp_path, second)
    resolved = resolve_latest_pointer(tmp_path)
    assert resolved is not None
    assert resolved.resolve() == second.resolve()


def test_resolve_latest_pointer_returns_none_when_missing(tmp_path: Path) -> None:
    assert resolve_latest_pointer(tmp_path) is None


def test_latest_txt_fallback_is_read_when_no_symlink(tmp_path: Path) -> None:
    target = _make_target(tmp_path)
    (tmp_path / "latest.txt").write_text(str(target.resolve()), encoding="utf-8")
    resolved = resolve_latest_pointer(tmp_path)
    assert resolved is not None
    assert resolved.resolve() == target.resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only symlink path")
def test_symlink_path_on_posix(tmp_path: Path) -> None:
    target = _make_target(tmp_path)
    update_latest_pointer(tmp_path, target)
    link = tmp_path / "latest"
    assert link.is_symlink()


def test_fallback_uses_latest_txt_when_symlink_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force both symlink + junction to fail and assert latest.txt is written."""

    target = _make_target(tmp_path)

    def _fail_symlink(*args: object, **kwargs: object) -> None:
        raise OSError("symlink blocked")

    monkeypatch.setattr("pathlib.Path.symlink_to", _fail_symlink)
    # Also make the junction path inert by pretending we're not on Windows.
    monkeypatch.setattr("sys.platform", "linux")

    pointer = update_latest_pointer(tmp_path, target)
    assert pointer == tmp_path / "latest.txt"
    assert pointer.read_text(encoding="utf-8") == str(target.resolve())
    resolved = resolve_latest_pointer(tmp_path)
    assert resolved is not None
    assert resolved.resolve() == target.resolve()


def test_latest_pointer_does_not_clobber_real_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real ``latest/`` directory must not be removed — only pointers are."""

    real_dir = tmp_path / "latest"
    real_dir.mkdir()
    (real_dir / "real_file").write_text("preserve me", encoding="utf-8")

    target = _make_target(tmp_path)
    # Force symlink + junction paths to fail so the fallback runs; the
    # existing real directory should still be present afterward.
    monkeypatch.setattr(
        "pathlib.Path.symlink_to", lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))
    )
    monkeypatch.setattr("sys.platform", "linux")

    update_latest_pointer(tmp_path, target)

    assert real_dir.is_dir()
    assert (real_dir / "real_file").read_text(encoding="utf-8") == "preserve me"
    # The text-file pointer still went to latest.txt.
    assert (tmp_path / "latest.txt").is_file()


def test_resolve_latest_pointer_handles_missing_target_in_txt(tmp_path: Path) -> None:
    """``latest.txt`` can point at a directory that no longer exists."""

    (tmp_path / "latest.txt").write_text(str(tmp_path / "does-not-exist"), encoding="utf-8")
    assert resolve_latest_pointer(tmp_path) is None


def test_os_symlink_permission_error_does_not_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate Windows SeCreateSymbolicLinkPrivilege absence."""

    target = _make_target(tmp_path)

    def _winerror(*args: object, **kwargs: object) -> None:
        raise OSError(os.errno.EACCES if hasattr(os, "errno") else 13, "privilege not held")

    monkeypatch.setattr("pathlib.Path.symlink_to", _winerror)
    monkeypatch.setattr("sys.platform", "linux")  # disable junction path

    pointer = update_latest_pointer(tmp_path, target)
    assert pointer.name in ("latest", "latest.txt")
