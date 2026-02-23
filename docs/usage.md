# Usage Guide

## Install

```bash
pip install haema
```

Development:

```bash
pip install -e ".[dev]"
```

## Source of Truth

API docstrings in `haema/` are authoritative. This guide is a quick checklist
for implementing adapters and bootstrapping `Memory`.

## Minimal Setup

Provide two adapters:

- `EmbeddingClient`
- `LLMClient`

```python
from haema import EmbeddingClient, LLMClient, Memory


class MyEmbeddingClient(EmbeddingClient):
    ...


class MyLLMClient(LLMClient):
    ...


m = Memory(
    path="./haema_store",
    output_dimensionality=1536,
    embedding_client=MyEmbeddingClient(),
    llm_client=MyLLMClient(),
    merge_top_k=3,
    merge_distance_cutoff=0.25,
)
```

## Embedding Adapter Checklist

Implement:

- `embed_query(texts, output_dimensionality)`
- `embed_document(texts, output_dimensionality)`

Contract:

- return a 2D `numpy.ndarray`
- dtype must be `float32`
- shape must be `(len(texts), output_dimensionality)`
- keep query/document task settings separated when supported by provider

## LLM Adapter Checklist

Implement:

- `generate_structured(system_prompt, user_prompt, response_model) -> dict[str, Any]`

Contract:

- return a dict parseable by `response_model.model_validate(...)`
- propagate provider failures as exceptions
- do not return free-form text when structured output is required

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
