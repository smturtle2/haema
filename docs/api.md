# API Reference

`haema` exposes three top-level public contracts:

- `Memory`
- `EmbeddingClient`
- `LLMClient`

Code docstrings in `haema/` are the source of truth. This page mirrors those
contracts for quick scanning.

## Memory

### Constructor

`Memory(path, output_dimensionality, embedding_client, llm_client, merge_top_k=3, merge_distance_cutoff=0.25)`

Args:

- `path: str | Path`: storage root directory
- `output_dimensionality: int`: required embedding dimension
- `embedding_client: EmbeddingClient`: embedding adapter
- `llm_client: LLMClient`: structured-output adapter
- `merge_top_k: int`: related candidate count per input
- `merge_distance_cutoff: float`: distance threshold for related-memory merge

Raises:

- `ImportError`: `chromadb` is not installed
- `ValueError`: invalid numeric options

Behavior:

- Creates and manages:
  - `<path>/db`
  - `<path>/core.md`
  - `<path>/latest.sqlite3`

### `get_core() -> str`

Returns:

- full `<path>/core.md` markdown text

Behavior:

- always reads the latest file content from disk

### `get_latest(begin: int, count: int) -> list[str]`

Args:

- `begin`: 1-indexed start position (`1` means newest)
- `count`: max number of items to return

Returns:

- latest memory documents sorted by descending `timestamp_ms`

Raises:

- `ValueError`: when `begin < 1`

Behavior:

- `count <= 0` returns `[]`
- out-of-range `begin` returns `[]`

### `search(content: str, n: int) -> list[str]`

Args:

- `content`: semantic query text
- `n`: max number of matches

Returns:

- top matched long-term memory documents

Behavior:

- embeds queries using `EmbeddingClient.embed_query(...)`
- `n <= 0` returns `[]`

### `add(contents: str | list[str]) -> None`

Args:

- `contents`: one raw text (`str`) or a batch of memory candidates (`list[str]`)

Returns:

- `None`

Raises:

- `TypeError`: list input contains non-string values
- `ValueError`: embedding shape/dimension contract mismatch
- provider/database exceptions are propagated

Behavior:

1. Normalize incoming contents.
2. Find related long-term memories.
3. Reconstruct merged memories with one refinement retry on incomplete output.
4. Upsert new memories and then delete replaced related memories.
5. Update core memory once for the call.

## EmbeddingClient

### `embed_query(texts, output_dimensionality) -> np.ndarray`

Args:

- `texts`: query strings as one batch
- `output_dimensionality`: embedding width

Returns:

- 2D `numpy.ndarray`
- dtype `float32`
- shape `(len(texts), output_dimensionality)`

### `embed_document(texts, output_dimensionality) -> np.ndarray`

Args:

- `texts`: document strings as one batch
- `output_dimensionality`: embedding width

Returns:

- 2D `numpy.ndarray`
- dtype `float32`
- shape `(len(texts), output_dimensionality)`

## LLMClient

### `generate_structured(system_prompt, user_prompt, response_model) -> dict[str, Any]`

Args:

- `system_prompt`: system instruction text
- `user_prompt`: user prompt text
- `response_model`: target Pydantic schema

Returns:

- dictionary parseable by `response_model.model_validate(...)`

Behavior:

- provider failures should be propagated as exceptions
