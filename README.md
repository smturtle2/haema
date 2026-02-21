# HAEMA

[English](README.md) | [한국어](README.ko.md)

HAEMA is a memory framework for agents built on ChromaDB.

It manages three memory modes behind one simple API:

- `core memory`: stable high-impact information (`get_core`)
- `latest memory`: recent memory slices by timestamp (`get_latest`)
- `long-term memory`: semantic search (`search`)

You only write through `add(contents)`. HAEMA updates all layers automatically.

## Features

- Single write API (`add`) for all memory layers
- ChromaDB-backed long-term memory (`collection="memory"`)
- Core memory kept as human-readable markdown (`core.md`)
- LLM/Embedding provider abstraction via client interfaces
- Automatic merge-and-replace behavior for related memories
- Conservative core updates with structured schema outputs

## Installation

```bash
pip install haema
```

Development install:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from haema import Memory

m = Memory(
    path="./haema_store",       # directory root; creates ./haema_store/db and ./haema_store/core.md
    output_dimensionality=1536,
    embedding_client=...,       # your EmbeddingClient implementation
    llm_client=...,             # your LLMClient implementation
)

m.add([
    "The user prefers concise and actionable responses.",
    "The user is building HAEMA on top of ChromaDB.",
])

print(m.get_core())                    # str
print(m.get_latest(begin=1, count=5)) # list[str]
print(m.search("user preference", 3))  # list[str]
```

For a real provider integration example, see:

- `examples/google_genai_example.py`

## Public API

### Constructor

`Memory(path, output_dimensionality, embedding_client, llm_client, merge_top_k=5, merge_distance_cutoff=0.25)`

- `path`: storage root directory
- `output_dimensionality`: embedding output dimension
- `embedding_client`: user-implemented embedding adapter
- `llm_client`: user-implemented structured LLM adapter
- `merge_top_k`: max candidates to consider as related memory
- `merge_distance_cutoff`: related-memory threshold for cosine distance

### Methods

- `get_core() -> str`
- `get_latest(begin: int, count: int) -> list[str]`
- `search(content: str, n: int) -> list[str]`
- `add(contents: list[str]) -> None`

## Client Interfaces

You implement both abstract clients:

- `EmbeddingClient.embed(texts, output_dimensionality) -> np.ndarray`
- `LLMClient.generate_structured(system_prompt, user_prompt, response_model) -> dict[str, Any]`

Embedding contract:

- return 2D `numpy.ndarray`
- dtype `float32`
- shape `(len(texts), output_dimensionality)`

LLM contract:

- return a JSON-like `dict` parseable by `response_model`
- HAEMA retries structured parsing up to 3 attempts
- fallback path is applied after repeated parsing failure

## Storage Layout

Given `path="./haema_store"`:

- long-term vector DB: `./haema_store/db`
- core markdown: `./haema_store/core.md`

Long-term memory metadata includes:

- `timestamp` (UTC ISO8601, e.g. `2026-02-21T03:20:00Z`)
- `timestamp_ms` (Unix epoch milliseconds)

## How `add()` Works

1. Normalize input strings and batch-embed contents
2. Query related memories with `top_k + distance cutoff`
3. If related memories exist:
   - synthesize updated memories via LLM
   - upsert synthesized memories
   - delete previous related memories
4. If no related memories exist:
   - reconstruct one or more memories from new content
   - upsert reconstructed memories
5. Update `core.md` once after processing all input contents

## Documentation

- `docs/index.md` - docs entrypoint
- `docs/usage.md` - usage and provider integration patterns
- `docs/api.md` - API behavior and edge cases
- `docs/architecture.md` - internal memory flow
- `docs/release.md` - GitHub Actions + PyPI release flow

## Release

PyPI publishing is automated by:

- `.github/workflows/publish-pypi.yml`

Trigger by pushing a tag like `v0.1.0`.

## License

MIT
