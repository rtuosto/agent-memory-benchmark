"""``results/latest`` pointer — symlink, Windows junction, or ``latest.txt``.

Tried in that order so the pointer works on the widest set of environments:

1. ``os.symlink(target, link)`` — works on POSIX + on Windows if developer
   mode or elevated privileges are available.
2. ``_winapi.CreateJunction`` — Windows-only fallback that doesn't require
   any special privileges; looks and acts like a directory symlink.
3. ``latest.txt`` — a plain text file containing the absolute path of the
   target. The CLI / docs read this if ``latest`` is absent.

Why all three: the predecessor was burned by Windows CI runners where the
symlink path throws ``OSError(1314)`` (privilege not held). Junctions
survive that case. Everything else falls through to ``latest.txt`` rather
than failing the run outright — the whole run directory is written before
the pointer update, so losing the pointer is never data loss.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger(__name__)


def _path_is_junction(path: Path) -> bool:
    """Best-effort junction test. ``Path.is_junction`` was added in 3.12."""

    fn = getattr(path, "is_junction", None)
    if callable(fn):
        try:
            return bool(fn())
        except OSError:
            return False
    return False


def _clear_existing(link: Path) -> None:
    """Remove ``link`` if it is a pointer, leave it alone if it's a real dir."""

    if link.is_symlink() or _path_is_junction(link):
        try:
            link.unlink()
        except OSError as e:
            _log.debug("could not unlink existing pointer %s: %s", link, e)
        return
    if link.exists() and link.is_file():
        try:
            link.unlink()
        except OSError as e:
            _log.debug("could not unlink stale file %s: %s", link, e)
        return
    if link.exists() and link.is_dir():
        _log.warning(
            "Refusing to remove %s: expected pointer but found a real directory.",
            link,
        )


def _try_symlink(link: Path, target: Path, results_base: Path) -> bool:
    try:
        rel = os.path.relpath(target, results_base)
        link.symlink_to(rel, target_is_directory=True)
        return True
    except OSError as e:
        _log.debug("symlink for latest pointer failed: %s", e)
        return False


def _try_junction(link: Path, target: Path) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import _winapi

        _winapi.CreateJunction(str(target.resolve()), str(link.resolve()))
        return True
    except (OSError, AttributeError, ImportError) as e:
        _log.debug("junction for latest pointer failed: %s", e)
        return False


def update_latest_pointer(results_base: Path, target: Path) -> Path:
    """Point ``results_base/latest`` at ``target``.

    Returns the path that was written — either the ``latest`` pointer
    itself, or the ``latest.txt`` fallback. The caller gets enough info
    to log what happened without reaching into the internals.
    """

    results_base.mkdir(parents=True, exist_ok=True)
    link = results_base / "latest"
    txt_fallback = results_base / "latest.txt"

    _clear_existing(link)

    if _try_symlink(link, target, results_base):
        _remove_stale_fallback(txt_fallback)
        return link
    if _try_junction(link, target):
        _remove_stale_fallback(txt_fallback)
        return link

    _log.warning("Falling back to latest.txt for %s (no symlink / junction).", results_base)
    txt_fallback.write_text(str(target.resolve()), encoding="utf-8")
    return txt_fallback


def _remove_stale_fallback(path: Path) -> None:
    if path.is_file():
        try:
            path.unlink()
        except OSError as e:
            _log.debug("could not remove stale latest.txt: %s", e)


def resolve_latest_pointer(results_base: Path) -> Path | None:
    """Inverse of :func:`update_latest_pointer` — return the pointed-at dir.

    Checks the symlink/junction first, then the ``latest.txt`` fallback.
    Returns ``None`` if nothing points anywhere.
    """

    link = results_base / "latest"
    if link.exists() and link.is_dir():
        return link.resolve()
    txt = results_base / "latest.txt"
    if txt.is_file():
        pointed = Path(txt.read_text(encoding="utf-8").strip())
        if pointed.exists():
            return pointed
    return None


__all__ = ["resolve_latest_pointer", "update_latest_pointer"]
