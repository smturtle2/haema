# Usage Guide

## Install

```bash
pip install haema
```

Development:

```bash
pip install -e ".[dev]"
```

## Minimal Setup

Provide two adapters:

- `EmbeddingClient`
- `LLMClient`

```python
from haema import Memory

m = Memory(
    path="./haema_store",
    output_dimensionality=1536,
    embedding_client=...,
    llm_client=...,
    merge_top_k=3,
    merge_distance_cutoff=0.25,
)
```

## Embedding Adapter Requirements

Implement:

- `embed_query(texts, output_dimensionality)`
- `embed_document(texts, output_dimensionality)`

Each must return:

- 2D `numpy.ndarray`
- dtype `float32`
- shape `(len(texts), output_dimensionality)`

## LLM Adapter Requirements

Implement:

- `generate_structured(system_prompt, user_prompt, response_model) -> dict[str, Any]`

Must return a dict parseable by the provided Pydantic model.

## Google GenAI Example

See:

- `examples/google_genai_example.py`

Run:

```bash
export GOOGLE_API_KEY="YOUR_KEY"
python3 examples/google_genai_example.py
```

The example uses:

- query embedding task type: `RETRIEVAL_QUERY`
- document embedding task type: `RETRIEVAL_DOCUMENT`

## Storage Layout

Given `path="./haema_store"`:

- vector DB: `./haema_store/db`
- core markdown: `./haema_store/core.md`
- latest index DB: `./haema_store/latest.sqlite3`
