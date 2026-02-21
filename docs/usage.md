# Usage Guide

## Install

```bash
pip install haema
```

For development:

```bash
pip install -e ".[dev]"
```

## Minimal Setup

You must provide two clients:

- `EmbeddingClient`
- `LLMClient`

```python
from haema import Memory

m = Memory(
    path="./haema_store",
    output_dimensionality=1536,
    embedding_client=...,
    llm_client=...,
)
```

## Google GenAI Example

See:

- `examples/google_genai_example.py`

Run:

```bash
export GOOGLE_API_KEY="YOUR_KEY"
python3 examples/google_genai_example.py
```

## Expected Embedding Return Shape

`EmbeddingClient.embed(...)` must return:

- 2D `numpy.ndarray`
- dtype `float32`
- shape `(len(texts), output_dimensionality)`

## Expected LLM Structured Return

`LLMClient.generate_structured(...)` must return:

- `dict[str, Any]`
- parseable by the provided `response_model`

## Storage Directory

Given `path="./haema_store"`:

- vector DB: `./haema_store/db`
- core markdown: `./haema_store/core.md`
