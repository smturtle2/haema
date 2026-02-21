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
