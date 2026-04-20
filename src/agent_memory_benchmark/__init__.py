"""agent-memory-benchmark — measurement instrument for agent memory systems.

This package evaluates memory systems against benchmark datasets
(LongMemEval, LOCOMO, BEAM) and produces reproducible scorecards covering
memory quality, wall-time performance, retrieval footprint, and retrieval
quality against evidence annotations.

The benchmark calls into memory systems through a transport-neutral adapter
layer. Memory systems have zero dependency on this package. An optional
compatibility ``Protocol`` (``MemorySystemShape``) is published as a spec
memory-system authors may choose to match for zero-glue benchmarking.
"""

from __future__ import annotations

from .version import __version__

__all__ = ["__version__"]
