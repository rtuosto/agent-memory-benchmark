"""BEAM loader — HF ``Mohammadta/BEAM`` + ``Mohammadta/BEAM-10M`` at a pinned revision.

BEAM (*Beyond a Million Tokens*, `arxiv 2510.27246
<https://arxiv.org/abs/2510.27246>`_) probes agent memory systems with
128K–1M token conversations (``Mohammadta/BEAM``) or 10M-extended
conversations (``Mohammadta/BEAM-10M``). The dataset ships
**conversation-per-row**, not question-per-row — one HF row contains
the full multi-session chat plus a ``probing_questions`` JSON bundle
keyed by the ten-way BEAM ability taxonomy
(:data:`CANONICAL_ABILITIES`).

Row → BenchmarkCase shape:

- ``chat`` is a list of session-lists (3 sessions on the 100K/500K/1M
  tiers). Each turn dict carries ``content``, ``role``, ``id``
  (globally unique ``int`` across the conversation's chat), and
  ``time_anchor`` (populated on the first turn of each session). Turn
  IDs are stringified ``id`` values so ``source_chat_ids`` maps
  directly to ``evidence_turn_ids``.
- ``probing_questions`` is a JSON-encoded string (sometimes Python-repr
  rather than strict JSON, so we parse with
  :func:`ast.literal_eval` as a fallback). Keys are BEAM's underscore-
  style ability names (``temporal_reasoning``, ``event_ordering``, …);
  we normalize to the hyphenated form the benchmark uses elsewhere.
- Each probing question becomes one :class:`QAItem` appended to the
  case's ``qa`` tuple. Gold answer is read from whichever of ``answer``
  / ``ideal_response`` / ``ideal_summary`` is populated. The
  ``source_chat_ids`` field (list of int ids or dict of such lists)
  flattens to ``evidence_turn_ids`` as stringified ids.

Revision pinning: ``HF_REVISION`` defaults to ``"main"`` as a visible
placeholder — override via the ``revision`` kwarg for publishable
runs until the canonical pin lands here. The descriptor hash includes
the revision string, so results across revisions are not cache-
compatible.

``datasets`` is imported lazily inside :func:`_load_hf` so this
module stays importable on environments where the HF toolchain cannot
be loaded — tests exercise row conversion with hand-built rows.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterator, Sequence
from typing import Any

from ..types import BenchmarkCase, DatasetName, QAItem, Session, Turn
from .base import DatasetAdapter

HF_DATASET_ID = "Mohammadta/BEAM"
HF_DATASET_ID_10M = "Mohammadta/BEAM-10M"
HF_REVISION = "main"
"""Default HF revision. Placeholder — override via ``revision=`` until a
canonical commit SHA is recorded in this module."""


# BEAM's ten-way memory-ability taxonomy. Names are hyphenated-lowercase
# to match LongMemEval's convention; the loader normalizes the HF-side
# underscore form (``temporal_reasoning``) to these values before
# storing on ``QAItem.question_type``.
CANONICAL_ABILITIES: tuple[str, ...] = (
    "abstention",
    "contradiction-resolution",
    "event-ordering",
    "information-extraction",
    "instruction-following",
    "knowledge-update",
    "multi-session-reasoning",
    "preference-following",
    "summarization",
    "temporal-reasoning",
)

VALID_VARIANTS: tuple[str, ...] = ("beam", "beam-10m")

# HF splits on BEAM are context-length tiers, not train/val/test. Pick
# the largest tier per variant by default so the full-context baseline
# actually stretches the instrument.
VALID_SPLITS: dict[str, tuple[str, ...]] = {
    "beam": ("100K", "500K", "1M"),
    "beam-10m": ("1M", "5M", "10M"),
}

_DEFAULT_SPLIT: dict[str, str] = {"beam": "1M", "beam-10m": "10M"}

_VARIANT_TO_DATASET: dict[str, str] = {
    "beam": HF_DATASET_ID,
    "beam-10m": HF_DATASET_ID_10M,
}


def _ability_from_raw(raw: str) -> str:
    """Normalize ``temporal_reasoning`` → ``temporal-reasoning``."""

    return raw.strip().lower().replace("_", "-")


class BeamDataset(DatasetAdapter):
    """BEAM conversations as a stream of :class:`BenchmarkCase` instances.

    One HF row → one :class:`BenchmarkCase` with N :class:`QAItem`
    instances (one per probing question across all abilities). The
    ``abilities`` filter gates which abilities contribute QA items to
    the emitted case; cases whose QA set is empty after filtering are
    still yielded (sessions remain available for ingestion).
    """

    name: DatasetName = "beam"

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        variant: str,
        revision: str,
        split: str = "",
        abilities: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> None:
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS!r}, got {variant!r}.")
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")

        ability_filter: frozenset[str] | None = None
        if abilities is not None:
            normalized = [_ability_from_raw(a) for a in abilities if str(a).strip()]
            unknown = [a for a in normalized if a not in CANONICAL_ABILITIES]
            if unknown:
                raise ValueError(
                    f"Unknown BEAM abilities: {unknown!r}. "
                    f"Valid choices: {list(CANONICAL_ABILITIES)}."
                )
            ability_filter = frozenset(normalized) if normalized else None

        materialized = list(rows)
        if limit is None or limit >= len(materialized):
            self._rows: list[dict[str, Any]] = materialized
            self._applied_limit_sig = "all"
        else:
            self._rows = materialized[:limit]
            self._applied_limit_sig = f"head:{limit}"

        self._variant = variant
        self._revision = revision
        self._split = split
        self._limit = limit
        self._ability_filter = ability_filter
        self._abilities_sig = "all" if ability_filter is None else ",".join(sorted(ability_filter))

    @classmethod
    def load(
        cls,
        *,
        variant: str = "beam",
        revision: str = HF_REVISION,
        split: str | None = None,
        abilities: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> BeamDataset:
        """Pull the dataset from HF at ``revision`` and wrap it."""

        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS!r}, got {variant!r}.")
        effective_split = split if split is not None else _DEFAULT_SPLIT[variant]
        allowed = VALID_SPLITS[variant]
        if effective_split not in allowed:
            raise ValueError(
                f"BEAM variant {variant!r} only supports splits {list(allowed)}, "
                f"got {effective_split!r}."
            )
        rows = _load_hf(variant=variant, revision=revision, split=effective_split)
        return cls(
            rows,
            variant=variant,
            revision=revision,
            split=effective_split,
            abilities=abilities,
            limit=limit,
        )

    def __iter__(self) -> Iterator[BenchmarkCase]:
        for index, row in enumerate(self._rows):
            yield _row_to_case(row, index=index, ability_filter=self._ability_filter)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def split(self) -> str:
        return self._split

    def descriptor_hash(self) -> str:
        parts = (
            self.name,
            self._variant,
            self._revision,
            self._split,
            self._abilities_sig,
            self._applied_limit_sig,
        )
        payload = b"\x1e".join(p.encode("utf-8") for p in parts)
        return hashlib.sha256(payload).hexdigest()


def load_beam(
    *,
    variant: str = "beam",
    revision: str = HF_REVISION,
    split: str | None = None,
    abilities: Sequence[str] | None = None,
    limit: int | None = None,
) -> BeamDataset:
    """Public entry point. Wraps :meth:`BeamDataset.load`."""

    return BeamDataset.load(
        variant=variant,
        revision=revision,
        split=split,
        abilities=abilities,
        limit=limit,
    )


def _row_to_case(
    row: dict[str, Any],
    *,
    index: int,
    ability_filter: frozenset[str] | None = None,
) -> BenchmarkCase:
    """Convert one BEAM HF row into a :class:`BenchmarkCase`."""

    case_id = str(row.get("conversation_id") or f"beam_{index}")
    sessions = _parse_chat(row.get("chat") or [])
    qa_items = _parse_probing_questions(
        row.get("probing_questions"),
        case_id=case_id,
        ability_filter=ability_filter,
    )
    return BenchmarkCase(
        case_id=case_id,
        sessions=tuple(sessions),
        qa=tuple(qa_items),
        dataset="beam",
    )


def _parse_chat(chat: list[Any]) -> list[Session]:
    """Flatten the nested ``chat`` list into :class:`Session` objects.

    Each top-level entry in ``chat`` is a session (a list of turn
    dicts). Turns carry a globally-unique ``id`` int; we stringify it
    so ``source_chat_ids`` evidence references match directly. The
    first turn's ``time_anchor`` becomes ``session_time``.
    """

    sessions: list[Session] = []
    for session_idx, raw_session in enumerate(chat, start=1):
        if not isinstance(raw_session, list):
            continue
        turns: list[Turn] = []
        session_time: str | None = None
        for raw_turn in raw_session:
            if not isinstance(raw_turn, dict):
                continue
            role = raw_turn.get("role")
            content = raw_turn.get("content")
            tid = raw_turn.get("id")
            if role is None or content is None or tid is None:
                continue
            if session_time is None:
                ta = raw_turn.get("time_anchor")
                if ta:
                    session_time = str(ta)
            turns.append(
                Turn(
                    turn_id=str(tid),
                    speaker=str(role),
                    text=str(content),
                    timestamp=_optional_str(raw_turn.get("time_anchor")),
                )
            )
        sessions.append(
            Session(
                session_index=session_idx,
                turns=tuple(turns),
                session_time=session_time,
                session_id=f"session_{session_idx}",
            )
        )
    return sessions


def _parse_probing_questions(
    raw: object,
    *,
    case_id: str,
    ability_filter: frozenset[str] | None,
) -> list[QAItem]:
    """Parse ``probing_questions`` into one :class:`QAItem` per question.

    ``probing_questions`` on HF is a string that's usually a Python
    repr (single-quoted) rather than strict JSON. We try both parsers
    before giving up — an unparseable field yields zero QA items
    (the row's sessions remain available for ingestion).
    """

    bundle = _parse_questions_bundle(raw)
    if bundle is None:
        return []

    items: list[QAItem] = []
    for ability_raw in sorted(bundle):
        ability = _ability_from_raw(ability_raw)
        if ability not in CANONICAL_ABILITIES:
            # Unknown ability; skip silently so a new BEAM release
            # doesn't hard-error the loader. Logged via metadata on
            # the row if the caller ever needs it.
            continue
        if ability_filter is not None and ability not in ability_filter:
            continue
        entries = bundle[ability_raw]
        if not isinstance(entries, list):
            continue
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            question = str(entry.get("question", "")).strip()
            if not question:
                continue
            gold = _gold_from_entry(entry)
            evidence_ids = _evidence_from_entry(entry)
            difficulty = entry.get("difficulty")
            metadata: dict[str, str] = {"ability": ability}
            if difficulty is not None:
                metadata["difficulty"] = str(difficulty)
            items.append(
                QAItem(
                    question_id=f"{case_id}:{ability}:{idx}",
                    question=question,
                    gold=gold,
                    question_type=ability,
                    evidence_turn_ids=evidence_ids,
                    metadata=metadata,
                )
            )
    return items


def _parse_questions_bundle(raw: object) -> dict[str, Any] | None:
    """Try JSON then Python-literal parsing. Returns ``None`` on failure."""

    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _gold_from_entry(entry: dict[str, Any]) -> str:
    """Pick the gold answer string from whichever field is populated.

    Priority: ``answer`` (most abilities) → ``ideal_response``
    (abstention) → ``ideal_summary`` (summarization) → empty string.
    """

    for key in ("answer", "ideal_response", "ideal_summary"):
        value = entry.get(key)
        if value:
            return str(value).strip()
    return ""


def _evidence_from_entry(entry: dict[str, Any]) -> tuple[str, ...]:
    """Extract ``source_chat_ids`` as stringified turn IDs.

    BEAM stores this as either a list of ints (most abilities) or a
    dict of such lists (``knowledge_update`` uses
    ``{"original_info": [...], "updated_info": [...]}``). ``None``
    means no evidence annotation for this question — the evidence KPIs
    will report ``null`` for it downstream.
    """

    raw = entry.get("source_chat_ids")
    if raw is None:
        return ()
    if isinstance(raw, list):
        return tuple(str(x) for x in raw if isinstance(x, (int, str)))
    if isinstance(raw, dict):
        out: list[str] = []
        for value in raw.values():
            if isinstance(value, list):
                out.extend(str(x) for x in value if isinstance(x, (int, str)))
        return tuple(out)
    return ()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _load_hf(*, variant: str, revision: str, split: str) -> list[dict[str, Any]]:
    """Pull the dataset rows via the ``datasets`` library.

    Lazy import so this module stays importable on environments where
    the HF stack can't be loaded — matches the LongMemEval loader.
    """

    from datasets import load_dataset  # type: ignore[import-untyped]

    repo = _VARIANT_TO_DATASET[variant]
    ds = load_dataset(repo, revision=revision, split=split)
    return list(ds)


__all__ = [
    "CANONICAL_ABILITIES",
    "HF_DATASET_ID",
    "HF_DATASET_ID_10M",
    "HF_REVISION",
    "VALID_SPLITS",
    "VALID_VARIANTS",
    "BeamDataset",
    "load_beam",
]
