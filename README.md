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
from haema import EmbeddingClient, LLMClient, Memory


class MyEmbeddingClient(EmbeddingClient):
    ...


class MyLLMClient(LLMClient):
    ...

m = Memory(
    path="./haema_store",               # storage root
    output_dimensionality=1536,         # embedding vector width
    embedding_client=MyEmbeddingClient(),
    llm_client=MyLLMClient(),
    merge_top_k=3,                      # related candidates per input
    merge_distance_cutoff=0.25,         # related-memory distance threshold
)

m.add([
    "The user prefers concise and actionable responses.",
    "The user is building HAEMA on top of ChromaDB.",
])

print(m.get_core())                      # str
print(m.get_latest(begin=1, count=5))   # list[str]
print(m.search("user preference", n=3)) # list[str]
```

Real provider example:

- `examples/google_genai_example.py`

## Public API

### Constructor

`Memory(path, output_dimensionality, embedding_client, llm_client, merge_top_k=3, merge_distance_cutoff=0.25)`

- `path: str | Path`: storage root directory
- `output_dimensionality: int`: required embedding dimension (`> 0`)
- `embedding_client: EmbeddingClient`: query/document embedding adapter
- `llm_client: LLMClient`: structured-output adapter
- `merge_top_k: int`: related candidate count per new content (default `3`, must be `> 0`)
- `merge_distance_cutoff: float`: related-memory distance threshold (default `0.25`, must be `>= 0`)

Validation:

- `output_dimensionality <= 0` -> `ValueError`
- `merge_top_k <= 0` -> `ValueError`
- `merge_distance_cutoff < 0` -> `ValueError`
- missing `chromadb` -> `ImportError`

### Methods

- `get_core() -> str`: returns full `<path>/core.md` text.
- `get_latest(begin: int, count: int) -> list[str]`: 1-indexed latest slice sorted by descending timestamp.
- `search(content: str, n: int) -> list[str]`: semantic search over long-term memory documents.
- `add(contents: str | list[str]) -> None`: single write API that updates long-term/latest/core layers.

Method behavior:

- `get_latest(begin < 1)` raises `ValueError`
- `get_latest(count <= 0)` returns `[]`
- `search(n <= 0)` returns `[]`
- `add(str)` runs pre-memory split first; `add(list[str])` uses normalized list items directly

## How To Implement Adapters

### `EmbeddingClient`

- `embed_query(texts, output_dimensionality) -> np.ndarray`
- `embed_document(texts, output_dimensionality) -> np.ndarray`

Checklist:

- return a 2D `numpy.ndarray`
- dtype must be `float32`
- shape must be `(len(texts), output_dimensionality)`
- keep query/document task settings separated when your provider supports it

### `LLMClient`

- `generate_structured(system_prompt, user_prompt, response_model) -> dict[str, Any]`

Checklist:

- return a `dict[str, Any]` parseable by `response_model.model_validate(...)`
- propagate provider failures as exceptions
- avoid returning unstructured free-form text

## Reconstruction Schema

HAEMA uses structured reconstruction output for long-term memory updates:

```python
class MemoryReconstructionResponse(BaseModel):
    memories: list[str]
    coverage: Literal["complete", "incomplete"]
```

If output is empty or `coverage == "incomplete"`, HAEMA runs one refinement pass.
If it still fails, HAEMA safely falls back to normalized `contents`.

## Prompt Contracts (Layer Responsibility)

HAEMA uses three independent prompt stages with separate outputs:

- pre-memory split:
  - input: one raw add string
  - output schema: `PreMemorySplitResponse(contents)`
  - responsibility: split factual units only (no core policy decision)
- reconstruction:
  - input: related memories + new contents
  - output schema: `MemoryReconstructionResponse(memories, coverage)`
  - responsibility: generate long-term memories only
- core update:
  - input: current core + reconstructed new memories
  - output schema: `CoreUpdateResponse(should_update, core_markdown)`
  - responsibility: conservative core update only

Prompt user blocks are boundary-labeled with tags such as:

- `<raw_input> ... </raw_input>`
- `<related_memories> ... </related_memories>`
- `<new_contents> ... </new_contents>`
- `<current_core_markdown> ... </current_core_markdown>`
- `<candidate_new_memories> ... </candidate_new_memories>`

These tags are prompt-boundary markers for model clarity, not parser/runtime control logic.

## Core Memory Policy

Core memory should keep only durable, high-impact, high-confidence information.
By prompt policy, candidate items should pass:

1. durability across sessions
2. material impact on future agent behavior
3. high confidence grounded in evidence

Core prompt policy also enforces:

- strict section routing to one of `SOUL/TOOLS/RULE/USER`
- exclusion of temporary/session-only/transient logs and noise
- compact high-signal output with a soft target budget around 8 bullets total

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
