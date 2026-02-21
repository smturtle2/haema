# API Reference

## Constructor

`Memory(path, output_dimensionality, embedding_client, llm_client, merge_top_k=3, merge_distance_cutoff=0.25)`

Parameters:

- `path: str | Path`
- `output_dimensionality: int`
- `embedding_client: EmbeddingClient`
- `llm_client: LLMClient`
- `merge_top_k: int` (default `3`)
- `merge_distance_cutoff: float` (default `0.25`)

Validation:

- `output_dimensionality > 0`
- `merge_top_k > 0`
- `merge_distance_cutoff >= 0`

## Methods

### `get_core() -> str`

Returns full core markdown from `<path>/core.md`.

### `get_latest(begin: int, count: int) -> list[str]`

- 1-indexed (`begin=1` is most recent)
- sorted by `timestamp_ms` descending
- returns only text

Behavior:

- `begin < 1` raises `ValueError`
- `count <= 0` returns `[]`
- out-of-range begin returns `[]`

### `search(content: str, n: int) -> list[str]`

- embeds `content` with `embed_query`
- queries top `n` from Chroma
- returns matched document texts only

Behavior:

- `n <= 0` returns `[]`

### `add(contents: str | list[str]) -> None`

Single-pass N:M reconstruction pipeline:

1. Normalize and trim `contents`.
   - if a single `str` is passed, HAEMA expands it into pre-memory items via structured LLM output
2. Batch query-embed all normalized contents.
3. For each embedding, query top-k and keep `distance <= merge_distance_cutoff`.
4. Union related memories by ID.
5. Run one structured reconstruction call with:
   - related documents (may be empty)
   - all new contents
6. If output is empty or incomplete, run one refinement call.
7. If still invalid, fallback to normalized contents.
8. Upsert reconstructed memories with document embeddings.
9. Delete replaced related IDs after upsert.
10. Update core once.

## Client Interface Contracts

### `EmbeddingClient`

- `embed_query(texts, output_dimensionality) -> np.ndarray`
- `embed_document(texts, output_dimensionality) -> np.ndarray`

Return contract:

- 2D float32 array
- shape `(len(texts), output_dimensionality)`

### `LLMClient`

- `generate_structured(system_prompt, user_prompt, response_model) -> dict[str, Any]`

Must return a dict parseable by `response_model`.

## Structured Schemas

### `MemoryReconstructionResponse`

- `memories: list[str]`
- `coverage: Literal["complete", "incomplete"]`

### `CoreUpdateResponse`

- `should_update: bool`
- `core_markdown: Optional[str]`

## Error Model

- Embedding shape mismatch: `ValueError`
- Chroma operation failures: propagated
- Retryable structured LLM failures: up to 3 retries per call
- Reconstruction quality retry: 1 additional structured call
- Non-retryable LLM implementation errors: propagated
