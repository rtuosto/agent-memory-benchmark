# BEAM — *Beyond a Million Tokens*

BEAM ([arxiv 2510.27246](https://arxiv.org/abs/2510.27246)) is the
third dataset supported by `agent-memory-benchmark`. It probes agent
memory systems against 2000 questions × 100 conversations × 10 memory
abilities, with conversation context windows ranging from 128K to 1M
tokens (and up to 10M on the `beam-10m` variant).

BEAM's direction is orthogonal to LongMemEval (broad question types,
100–150K-token haystacks) and LOCOMO (10 long-form conversations with
10-way majority-vote judging). Use it to stress the retrieval layer at
context lengths beyond what either of the other datasets exercises.

## Variants + splits

| Variant | HF repo | Context tiers (splits) |
|---|---|---|
| `beam` | [`Mohammadta/BEAM`](https://huggingface.co/datasets/Mohammadta/BEAM) | `100K`, `500K`, `1M` |
| `beam-10m` | [`Mohammadta/BEAM-10M`](https://huggingface.co/datasets/Mohammadta/BEAM-10M) | `1M`, `5M`, `10M` |

Splits on BEAM are **context-length tiers**, not train/val/test. The
loader defaults to the largest tier for the chosen variant (`1M` for
`beam`, `10M` for `beam-10m`) so the full-context baseline actually
stretches the instrument. Override with `--split 100K` when you want
to iterate faster during development.

## Ability taxonomy

The benchmark pins the ten canonical memory abilities observed on
`Mohammadta/BEAM` (`CANONICAL_ABILITIES` in
`src/agent_memory_benchmark/datasets/beam.py`):

1. `abstention`
2. `contradiction-resolution`
3. `event-ordering`
4. `information-extraction`
5. `instruction-following`
6. `knowledge-update`
7. `multi-session-reasoning`
8. `preference-following`
9. `summarization`
10. `temporal-reasoning`

BEAM's HF rows use underscored names (`temporal_reasoning`,
`event_ordering`, …); the loader normalizes to the hyphenated form so
the scorecard's `per_category` bucketing matches LongMemEval's naming
convention. `--abilities a,b,c` accepts either form (`temporal_reasoning`
or `temporal-reasoning`); typos hard-error at load before a run burns
a judge-model bill.

## Ability → judge template routing

BEAM ships **four** byte-frozen judge templates
(`src/agent_memory_benchmark/judge/beam.py`):

| Template | Fingerprint | Ability routing |
|---|---|---|
| `general` | `4d2ba49c…048d010` | fallback for most abilities |
| `temporal` | `80601d4a…c6f7d2f` | `temporal-reasoning` |
| `event-ordering` | `6fca7bb1…d5588e7` | `event-ordering` |
| `abstention` | `00ed31aa…c10ccd6f` | `abstention` |

Bundle fingerprint (combined):
`671c56e0e99d40c92c6dcdb557b688e0323fc98fa7fa43b626ea69fb634c70a4`.

Abilities outside the specialized map (knowledge-update,
multi-hop-reasoning, preference-following,
information-integration, long-range-understanding,
selective-recall, persona-consistency) route to `general`.

**Why a generic template for seven of ten abilities?** The plan's
direction is "start with a generic grader and specialize only when
accuracy on a specific ability is suspect." Adding a specialized
template is a calibration change — a bumped `protocol_version`, a
re-baselined fingerprint golden in
`tests/unit/test_judge_prompts_stable.py`, and a documented migration.
Treat ability-template additions as P8 events, not drive-by commits.

## Row schema (live, confirmed against `Mohammadta/BEAM`)

One HF row = one **conversation** (not one question). The loader
explodes it into one `BenchmarkCase` with N `QAItem`s (one per probing
question across all abilities, typically 20 = 2 × 10). Row fields used:

- `conversation_id` — becomes `BenchmarkCase.case_id`.
- `chat` — list of 3 session-lists on the 100K/500K/1M tiers. Each
  turn dict has `content`, `role`, `id` (globally unique `int` across
  the conversation's chat), and `time_anchor` (populated on the first
  turn of each session and reused as `Session.session_time`). Turn
  IDs are stringified `id` values so `source_chat_ids` evidence refs
  map 1:1.
- `probing_questions` — a Python-repr string (occasionally strict JSON)
  keyed by the ten ability names (underscored form). The loader tries
  `json.loads` first and falls back to `ast.literal_eval` — some HF
  rows aren't valid JSON. Each ability value is a list of question
  dicts; each dict becomes one `QAItem`. Gold answer field varies by
  ability: `answer` (most) / `ideal_response` (abstention) /
  `ideal_summary` (summarization). `source_chat_ids` becomes
  `evidence_turn_ids` (list of ints stringified, or a dict of such
  lists for `knowledge_update` — `original_info` + `updated_info`
  concatenated).

Rows that omit `conversation_id` derive `case_id = f"beam_{index}"`.
Questions with no usable gold text still surface (empty `gold=""`) so
the scorecard's count-based denominators stay honest.

If BEAM adds an eleventh ability in a later release, the loader drops
its questions silently (the row's remaining 10-ability content is
unaffected). Adding the new ability is a one-line edit to
`CANONICAL_ABILITIES`; adding a specialized judge template for it is
a P8 event (fingerprint re-baseline).

## Revision pinning

`HF_REVISION` defaults to `"main"` as a **visible placeholder**.
Publishable BEAM runs must pass `--beam-revision <sha>` explicitly
until a canonical pin lands in `datasets/beam.py`. The descriptor hash
includes the revision string verbatim — results produced against
different revisions are not cross-comparable.

## Invocation

```bash
# Smoke test against the hardest beam tier:
amb run beam \
    --memory full-context \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b \
    --limit 5

# Filter to two abilities, pin a specific revision:
amb run beam \
    --memory python:mypkg.mem:MyMem \
    --answer-model ollama:llama3.1:8b \
    --judge-model openai:gpt-4o-mini-2024-07-18 \
    --variant beam --split 500K \
    --abilities temporal-reasoning,abstention \
    --beam-revision <sha> \
    --limit 40

# 10M-token extended runs:
amb run beam \
    --memory http://localhost:8000 \
    --answer-model openai:gpt-4o-mini-2024-07-18 \
    --judge-model openai:gpt-4o-2024-08-06 \
    --variant beam-10m --split 10M
```

## Scorecard shape

BEAM questions land in per-category buckets keyed by ability name
(`per_category.temporal-reasoning`, `per_category.abstention`, …). The
scorecard aggregates across the ten buckets for `macro_accuracy`; use
`amb compare run_A/ run_B/` to surface per-ability regressions between
runs.
