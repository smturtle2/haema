# Architecture

## Memory Layers

HAEMA tracks three layers:

- `core`: stable identity/policy/user profile (`core.md`)
- `latest`: recency index (`latest.sqlite3`, served by `get_latest`)
- `long-term`: vectorized memory store (`collection="memory"`)

## N:M Reconstruction Flow in `add()`

For each `add(contents)` call:

1. Normalize all incoming strings.
2. Batch-embed normalized contents with `embed_query`.
3. Query related memories for each content embedding (`top_k`, cosine distance cutoff).
4. Union all related IDs/documents across the batch.
5. Run one LLM reconstruction call over:
   - unioned related documents
   - all new contents
6. If result is incomplete or empty, run one refinement call.
7. If still not acceptable, fallback to normalized contents.
8. Embed final memories via `embed_document` and upsert.
9. Delete replaced related IDs only after successful upsert.
10. Update core once with memories added in this call.

## Stage Boundaries

The three LLM-backed stages are independent and have different artifacts:

1. pre-memory split
- consumes one raw add input
- emits `contents` for retrieval/reconstruction preparation
- does not decide core policy

2. reconstruction
- consumes related memories + new contents
- emits long-term `memories` and `coverage`
- does not directly edit core

3. core update
- consumes current core + reconstructed new memories
- emits `should_update/core_markdown`
- governs durable core-only information

Only the reconstruction output (`new_memories`) is passed into core update.

## Why One Reconstruction Pass

- Reduces repeated LLM calls for multi-content adds.
- Allows cross-content consolidation in one output set.
- Prevents per-content race-like overwrite behavior.

## Replacement Safety

Replacement order:

1. upsert new memories
2. delete old related memories

This minimizes accidental loss if upsert fails.

## Prompt Contract

Reconstruction prompt explicitly enforces:

- related memories may be empty
- new contents are newest evidence
- conflict resolution favors new contents
- independent facts should split into separate memories
- output schema strictness (`memories`, `coverage`)

Prompt input blocks are boundary-labeled with tags (`<raw_input>`, `<related_memories>`,
`<new_contents>`, `<current_core_markdown>`, `<candidate_new_memories>`).
These markers are used to clarify model input boundaries, not as parser-level runtime control.

## Core Entry Policy (Prompt-Level)

Core update prompt is intentionally conservative and requires coreworthiness:

1. durability across sessions
2. material impact on future behavior
3. high confidence from evidence

Items that fail the gate should be excluded from core. Prompt policy also asks for:

- strict one-section routing (`SOUL`, `TOOLS`, `RULE`, `USER`)
- exclusion of temporary/session-only/transient logs/noise
- compact high-signal core with a soft target around 8 total bullets
