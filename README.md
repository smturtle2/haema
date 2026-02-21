# HAEMA

[English](README.md) | [한국어](README.ko.md)

HAEMA is an agent memory framework built on ChromaDB.

It provides three memory modes through a single write API:

- `core memory`: durable high-impact identity/policy/user facts (`get_core`)
- `latest memory`: recency slice by timestamp (`get_latest`)
- `long-term memory`: semantic retrieval (`search`)

You only write through `add(contents)`, and HAEMA updates all layers automatically.

## Key Changes (Current)

- `add(contents)` runs a single N:M reconstruction pass per call.
- Embedding is split into query/document interfaces:
  - `embed_query(...)`
  - `embed_document(...)`
- no-related special path is removed; one reconstruction schema is used.
- reconstruction schema:
  - `memories: list[str]`
  - `coverage: "complete" | "incomplete"`

## Installation

```bash
pip install haema
```

Development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from haema import Memory

m = Memory(
    path="./haema_store",
    output_dimensionality=1536,
    embedding_client=...,   # your EmbeddingClient implementation
    llm_client=...,         # your LLMClient implementation
    merge_top_k=3,
    merge_distance_cutoff=0.25,
)

m.add([
    "The user prefers concise and actionable responses.",
    "The user is building HAEMA on top of ChromaDB.",
])

print(m.get_core())                    # str
print(m.get_latest(begin=1, count=5)) # list[str]
print(m.search("user preference", 3))  # list[str]
```

Real provider example:

- `examples/google_genai_example.py`

## Public API

### Constructor

`Memory(path, output_dimensionality, embedding_client, llm_client, merge_top_k=3, merge_distance_cutoff=0.25)`

- `path`: storage root directory
- `output_dimensionality`: embedding output dimension
- `embedding_client`: user embedding adapter
- `llm_client`: user structured-output LLM adapter
- `merge_top_k`: related candidate count per new content (default `3`)
- `merge_distance_cutoff`: related-memory distance threshold (default `0.25`)

### Methods

- `get_core() -> str`
- `get_latest(begin: int, count: int) -> list[str]`
- `search(content: str, n: int) -> list[str]`
- `add(contents: str | list[str]) -> None`

## Client Interfaces

### `EmbeddingClient`

- `embed_query(texts, output_dimensionality) -> np.ndarray`
- `embed_document(texts, output_dimensionality) -> np.ndarray`

Both must return:

- 2D `numpy.ndarray`
- dtype `float32`
- shape `(len(texts), output_dimensionality)`

### `LLMClient`

- `generate_structured(system_prompt, user_prompt, response_model) -> dict[str, Any]`

Must return a dict parseable by the provided Pydantic model.

## Reconstruction Schema

HAEMA uses structured reconstruction output for long-term memory updates:

```python
class MemoryReconstructionResponse(BaseModel):
    memories: list[str]
    coverage: Literal["complete", "incomplete"]
```

If output is empty or `coverage == "incomplete"`, HAEMA runs one refinement pass.
If it still fails, HAEMA safely falls back to normalized `contents`.

## Storage Layout

Given `path="./haema_store"`:

- long-term vector DB: `./haema_store/db`
- core markdown: `./haema_store/core.md`
- latest index DB: `./haema_store/latest.sqlite3`

Long-term metadata fields:

- `timestamp` (UTC ISO8601)
- `timestamp_ms` (Unix epoch milliseconds)

## How `add()` Works

1. Normalize input strings.
   - if `contents` is a single `str`, HAEMA first expands it into multiple pre-memory items via structured LLM output
2. Batch query-embed all `contents`.
3. For each query, fetch top-k and keep matches with distance cutoff.
4. Union related memories by `id`.
5. Run one reconstruction call with:
   - related memory documents (may be empty)
   - all new contents
6. Upsert reconstructed memories with document embeddings.
7. Delete replaced related IDs only after upsert succeeds.
8. Update core once per `add()` call.

## Breaking Changes

Compared to older builds:

1. `EmbeddingClient.embed(...)` is removed.
2. `NoRelatedMemoryResponse` is removed.
3. `MemorySynthesisResponse(update: list[str])` is replaced by `MemoryReconstructionResponse`.
4. `merge_top_k` default changed from `5` to `3`.

## Documentation

- `docs/index.md`
- `docs/usage.md`
- `docs/api.md`
- `docs/architecture.md`
- `docs/release.md`

## License

MIT
