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

Interpretation:

- `memories` must represent integrated long-term entries from related + new evidence.
- prioritize new evidence on conflict.
- keep entries atomic and non-duplicated.
- do not perform core editing in this stage.

- `coverage="complete"` only when important actionable facts from new contents are preserved.
- `coverage="incomplete"` when important facts are missing or over-compressed.

### `CoreUpdateResponse`

- `should_update: bool`
- `core_markdown: Optional[str]`

Interpretation:

- conservative default: keep `should_update=false` unless durable high-impact high-confidence updates exist.
- `core_markdown` must preserve section boundaries `SOUL/TOOLS/RULE/USER`.
- each bullet should belong to one section only.
- exclude transient/session-only low-signal information.
- keep core compact with a soft high-signal budget around 8 bullets.

## Prompt Boundary Contract

HAEMA prompt builders mark model input boundaries using tags:

- `<raw_input> ... </raw_input>`
- `<related_memories> ... </related_memories>`
- `<new_contents> ... </new_contents>`
- `<current_core_markdown> ... </current_core_markdown>`
- `<candidate_new_memories> ... </candidate_new_memories>`

These are prompt clarity markers, not runtime parser controls.

## Error Model

- Embedding shape mismatch: `ValueError`
- Chroma operation failures: propagated
- Retryable structured LLM failures: up to 3 retries per call
- Reconstruction quality retry: 1 additional structured call
- Non-retryable LLM implementation errors: propagated
