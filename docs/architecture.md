# Architecture

## Memory Layers

HAEMA tracks three memory layers:

- `core`: long-lived identity/policy profile (`core.md`)
- `latest`: recency slice from long-term storage (`get_latest`)
- `long-term`: vectorized memories in Chroma (`collection="memory"`)

## Data Flow in `add()`

For each incoming content:

1. Embed content.
2. Query top-k candidate memories.
3. Filter candidates by distance cutoff.
4. Branch:
   - related exists: synthesize + replace
   - no related: reconstruct fresh memory/memories
5. Store new memories with:
   - `timestamp` (ISO8601 UTC)
   - `timestamp_ms` (epoch ms)

After all contents:

6. Run one core update pass from accumulated newly added memories.

## Replace Semantics

On related-memory branch:

- new synthesized memories are upserted first
- old related memory IDs are deleted afterward

This order minimizes accidental data loss if upsert fails.

## Core Update Policy

Core update is intentionally conservative:

- only stable, high-impact information should be merged into core
- sections are fixed: `SOUL`, `TOOLS`, `RULE`, `USER`
- transient/session-only details should stay in latest/long-term memory
