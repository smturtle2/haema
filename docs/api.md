# API Reference

## Constructor

`Memory(path, output_dimensionality, embedding_client, llm_client, merge_top_k=5, merge_distance_cutoff=0.25)`

Parameters:

- `path: str | Path`
- `output_dimensionality: int`
- `embedding_client: EmbeddingClient`
- `llm_client: LLMClient`
- `merge_top_k: int` (default `5`)
- `merge_distance_cutoff: float` (default `0.25`)

Validation:

- `output_dimensionality > 0`
- `merge_top_k > 0`
- `merge_distance_cutoff >= 0`

## Methods

### `get_core() -> str`

Returns the full core markdown from `<path>/core.md`.

### `get_latest(begin: int, count: int) -> list[str]`

- 1-indexed (`begin=1` is most recent)
- sorts by `timestamp_ms` descending
- returns only memory text

Behavior:

- `begin < 1` raises `ValueError`
- `count <= 0` returns `[]`
- out-of-range begin returns `[]`

### `search(content: str, n: int) -> list[str]`

- embeds query text
- returns top `n` matched documents from Chroma
- no additional cutoff applied in `search`

Behavior:

- `n <= 0` returns `[]`

### `add(contents: list[str]) -> None`

Pipeline per content:

1. Trim/normalize content.
2. Find related memories (`top_k` + distance cutoff).
3. If related:
   - synthesize updated memories
   - upsert synthesized memories
   - delete old related memories
4. If no related:
   - reconstruct one or more durable memories
   - upsert reconstructed memories
5. After all contents:
   - update core once if needed

## Error Model

- Embedding shape mismatch raises `ValueError`.
- Chroma failures propagate.
- LLM structured parsing retries up to 3 attempts, then falls back.
