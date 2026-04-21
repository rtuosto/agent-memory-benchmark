"""``DatasetAdapter`` — the benchmark-facing dataset abstraction.

The runner iterates ``BenchmarkCase`` instances. Each adapter subclass knows
how to reach a specific dataset (LongMemEval on HF, LOCOMO from a local JSON
file, BEAM on HF) and is responsible for:

- producing :class:`BenchmarkCase` instances in a stable order,
- reporting ``len``,
- computing a ``descriptor_hash`` that ends up in the cache key so results
  are comparable only between byte-identical inputs.

The cache key recipe from PR-4 uses ``descriptor_hash`` as one of the SHA-256
inputs to both the ingestion and answer keys. Changing an adapter's
descriptor derivation therefore invalidates prior caches — treat it as a
migration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..types import BenchmarkCase, DatasetName


class DatasetAdapter(ABC):
    """One dataset's iteration + identity contract."""

    name: DatasetName

    @abstractmethod
    def __iter__(self) -> Iterator[BenchmarkCase]:
        """Yield :class:`BenchmarkCase` instances in a stable, reproducible order."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of cases this adapter will yield."""

    @abstractmethod
    def descriptor_hash(self) -> str:
        """Return the sha256 hex digest identifying this dataset view.

        Stable across processes and platforms for identical inputs. Flows
        into the cache key so two loaders that differ in split, revision,
        limit, or limit strategy produce non-overlapping caches.
        """


__all__ = ["DatasetAdapter"]
