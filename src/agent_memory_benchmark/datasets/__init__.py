"""Dataset adapters — one :class:`DatasetAdapter` per benchmark.

The runner holds a single ``DatasetAdapter`` and iterates it for
:class:`~agent_memory_benchmark.types.BenchmarkCase` instances. Each
adapter subclass knows how to source its dataset (HF-hosted, local JSON,
etc.) and how to derive a ``descriptor_hash`` that feeds the cache key.

BEAM lands in PR-11; ``load_dataset`` raises a descriptive error for it
until then.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import DatasetName
from .base import DatasetAdapter
from .locomo import LocomoDataset, load_locomo
from .longmemeval import (
    HF_DATASET_ID,
    HF_M_FILENAME,
    HF_REVISION,
    HF_S_FILENAME,
    LongMemEvalDataset,
    load_longmemeval,
)


class DatasetUnavailableError(Exception):
    """Raised when a dataset adapter is named but not yet implemented."""


def load_dataset(name: str, /, **kwargs: Any) -> DatasetAdapter:
    """Return a :class:`DatasetAdapter` for ``name``.

    Supported:

    - ``"longmemeval"`` — kwargs: ``split`` (required, ``"s"`` | ``"m"``),
      ``m_path``, ``revision``, ``limit``, ``limit_strategy``.
    - ``"locomo"`` — kwargs: ``path`` (required, local ``locomo10.json``),
      ``limit``.

    BEAM raises :class:`DatasetUnavailableError` until its loader lands.
    """

    normalized = name.strip().lower()
    if normalized == "longmemeval":
        split = kwargs.pop("split", None)
        if not split:
            raise ValueError("longmemeval requires a 'split' kwarg ('s' or 'm').")
        m_path = kwargs.pop("m_path", None)
        if m_path is not None and not isinstance(m_path, (str, Path)):
            raise TypeError(f"m_path must be str | Path, got {type(m_path).__name__}")
        return load_longmemeval(split, m_path=m_path, **kwargs)
    if normalized == "locomo":
        path = kwargs.pop("path", None)
        if path is None:
            raise ValueError(
                "locomo requires a 'path' kwarg — the local locomo10.json "
                "file path (LOCOMO is not redistributed via HF)."
            )
        if not isinstance(path, (str, Path)):
            raise TypeError(f"path must be str | Path, got {type(path).__name__}")
        return load_locomo(path, **kwargs)
    if normalized == "beam":
        raise DatasetUnavailableError("BEAM loader lands in PR-11.")
    raise ValueError(f"Unknown dataset {name!r}; expected one of: longmemeval, locomo, beam.")


__all__ = [
    "DatasetAdapter",
    "DatasetName",
    "DatasetUnavailableError",
    "HF_DATASET_ID",
    "HF_M_FILENAME",
    "HF_REVISION",
    "HF_S_FILENAME",
    "LocomoDataset",
    "LongMemEvalDataset",
    "load_dataset",
    "load_locomo",
    "load_longmemeval",
]
