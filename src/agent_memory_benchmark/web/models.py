"""Model + memory-adapter choice lists for the new-job form.

Answer/judge model specs come from two sources:

- **Ollama**: whatever ``ollama list`` currently exposes on this host.
  We shell out rather than parsing HTTP responses because the CLI is
  installed on every platform that has the daemon, and ``ollama list``
  is stable. Results are cached for 60s so opening the new-job form
  doesn't shell out once per page view.
- **OpenAI**: a short whitelist of models typically used in benchmark
  papers / judge pipelines. Curated, not exhaustive — the user can
  still type a custom ``openai:<model>`` directly since the template
  pairs each ``<select>`` with a free-text fallback.

Memory adapter presets are a small curated list. The full spec surface
(``python:pkg.mod:Cls``, ``http://...``, etc.) is too unbounded to
enumerate; the form keeps a free-text field alongside the dropdown so
custom specs still work.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

OPENAI_BENCHMARK_MODELS: tuple[str, ...] = (
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
    "openai:gpt-4-turbo",
    "openai:gpt-4",
    "openai:gpt-3.5-turbo",
    "openai:o1",
    "openai:o1-mini",
    "openai:o3-mini",
)

MEMORY_ADAPTER_PRESETS: tuple[str, ...] = (
    "full-context",
    "python:engram.memory:EngramGraphMemorySystem",
    "http://localhost:8001",
)

_OLLAMA_CACHE_TTL_S = 60.0


@dataclass
class _OllamaCache:
    fetched_at: float = 0.0
    models: tuple[str, ...] = ()


_ollama_cache = _OllamaCache()
_ollama_lock = threading.Lock()


def _parse_ollama_list(stdout: str) -> list[str]:
    """Extract model names from ``ollama list`` output.

    Expected layout::

        NAME                    ID              SIZE    MODIFIED
        llama3.1:8b             46e0c10c039e    4.9 GB  3 days ago

    We skip the header row (first line starting with ``NAME``) and
    take the first whitespace-delimited token on each subsequent line.
    Blank lines and stray header duplicates are ignored.
    """

    names: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("name"):
            continue
        first_tok = line.split(None, 1)[0]
        if first_tok:
            names.append(first_tok)
    return names


def _fetch_ollama_models() -> tuple[str, ...]:
    """Shell out to ``ollama list`` and return ``("ollama:<name>", ...)``.

    Failures (missing CLI, daemon down, non-zero exit) yield an empty
    tuple rather than raising — the new-job form still works with the
    OpenAI whitelist + free-text fallback.
    """

    cli = shutil.which("ollama")
    if cli is None:
        return ()
    try:
        result = subprocess.run(  # noqa: S603  # argv is fixed
            [cli, "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode != 0:
        return ()
    return tuple(f"ollama:{name}" for name in _parse_ollama_list(result.stdout))


def ollama_models(*, force_refresh: bool = False) -> tuple[str, ...]:
    """Return cached Ollama models, refreshing after ``_OLLAMA_CACHE_TTL_S``.

    The lock is held only while checking the cache timestamp — the
    actual shell-out happens outside the lock so a slow ``ollama list``
    doesn't block concurrent form opens on their cache-hit path.
    """

    now = time.monotonic()
    with _ollama_lock:
        if not force_refresh and now - _ollama_cache.fetched_at < _OLLAMA_CACHE_TTL_S:
            return _ollama_cache.models
    models = _fetch_ollama_models()
    with _ollama_lock:
        _ollama_cache.models = models
        _ollama_cache.fetched_at = time.monotonic()
    return models


def available_models() -> list[str]:
    """Combined ``ollama + openai`` model list for form dropdowns.

    Ollama models come first (they're the zero-cost default); OpenAI
    whitelist trails. Duplicates are not expected across providers
    since the prefix differs, but we de-dupe defensively.
    """

    seen: set[str] = set()
    out: list[str] = []
    for m in list(ollama_models()) + list(OPENAI_BENCHMARK_MODELS):
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def memory_adapter_presets() -> list[str]:
    """Return the built-in memory-adapter preset list.

    Kept as a function (rather than re-exporting the tuple) so a
    future config-driven version can slot in without API churn.
    """

    return list(MEMORY_ADAPTER_PRESETS)


__all__ = [
    "MEMORY_ADAPTER_PRESETS",
    "OPENAI_BENCHMARK_MODELS",
    "available_models",
    "memory_adapter_presets",
    "ollama_models",
]
