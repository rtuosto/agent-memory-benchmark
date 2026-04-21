"""Prompt-fingerprint utilities.

Every judge prompt template in this package is exposed as a module-level
``str`` constant. :func:`fingerprint` returns the SHA-256 hex digest of the
UTF-8 encoded template. Those digests are locked in
:mod:`tests.unit.test_judge_prompts_stable` — any inadvertent edit to a
template surfaces as a failing test with a precise diff of which prompt
drifted. Intentional changes require a re-baseline: update the golden
digest, bump the associated memory system's ``memory_version`` or
``protocol_version`` as appropriate, and document the migration.

This is a P8 ("benchmarking is a measurement instrument") invariant —
judge prompts are part of the instrument's calibration, not free
parameters an agent can tune per run.
"""

from __future__ import annotations

import hashlib


def fingerprint(template: str) -> str:
    """Return the SHA-256 hex digest of ``template`` encoded as UTF-8."""

    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def combined_fingerprint(templates: dict[str, str]) -> str:
    """Return the digest of ``templates`` treated as a single ordered bundle.

    Used for the per-benchmark ``judge_prompt_fingerprint`` that feeds the
    judge cache key (PR-4). Keys are sorted to make the derivation order-
    independent; the per-key digest is joined with an ASCII RS separator
    to match the cache-key framing convention.
    """

    parts: list[bytes] = []
    for key in sorted(templates):
        parts.append(key.encode("utf-8"))
        parts.append(fingerprint(templates[key]).encode("ascii"))
    return hashlib.sha256(b"\x1e".join(parts)).hexdigest()


__all__ = ["combined_fingerprint", "fingerprint"]
