from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from haema.clients import EmbeddingClient, LLMClient
from haema.prompts import (
    CORE_TEMPLATE,
    CORE_UPDATE_SYSTEM_PROMPT,
    NO_RELATED_MEMORY_SYSTEM_PROMPT,
    SYNTHESIZE_SYSTEM_PROMPT,
    build_core_update_user_prompt,
    build_no_related_memory_user_prompt,
    build_synthesize_user_prompt,
)
from haema.schemas import CoreUpdateResponse, MemorySynthesisResponse, NoRelatedMemoryResponse

try:
    import chromadb
except ImportError as exc:  # pragma: no cover - import error depends on runtime env
    chromadb = None
    _CHROMADB_IMPORT_ERROR = exc
else:
    _CHROMADB_IMPORT_ERROR = None

ModelT = TypeVar("ModelT", bound=BaseModel)


class Memory:
    COLLECTION_NAME = "memory"

    def __init__(
        self,
        path: str | Path,
        output_dimensionality: int,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
        merge_top_k: int = 5,
        merge_distance_cutoff: float = 0.25,
    ) -> None:
        if chromadb is None:
            raise ImportError("chromadb is required to use Memory") from _CHROMADB_IMPORT_ERROR
        if output_dimensionality <= 0:
            raise ValueError("output_dimensionality must be > 0")
        if merge_top_k <= 0:
            raise ValueError("merge_top_k must be > 0")
        if merge_distance_cutoff < 0:
            raise ValueError("merge_distance_cutoff must be >= 0")

        # `path` is a storage root directory. We store:
        # - long-term vector DB under `<path>/db`
        # - core memory markdown under `<path>/core.md`
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "db"
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.core_path = self.path / "core.md"
        if not self.core_path.exists():
            self.core_path.write_text(CORE_TEMPLATE, encoding="utf-8")

        self.output_dimensionality = output_dimensionality
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.merge_top_k = merge_top_k
        self.merge_distance_cutoff = merge_distance_cutoff
        self._last_timestamp_ms = 0

        client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def get_core(self) -> str:
        return self.core_path.read_text(encoding="utf-8")

    def get_latest(self, begin: int, count: int) -> list[str]:
        if begin < 1:
            raise ValueError("begin must be >= 1")
        if count <= 0:
            return []

        result = self.collection.get(include=["documents", "metadatas"])
        documents: list[str] = result.get("documents", []) or []
        metadatas: list[dict[str, Any] | None] = result.get("metadatas", []) or []

        pairs: list[tuple[str, int]] = []
        for idx, document in enumerate(documents):
            if not document:
                continue
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            timestamp_ms = self._extract_timestamp_ms(metadata)
            pairs.append((document, timestamp_ms))

        pairs.sort(key=lambda item: item[1], reverse=True)

        start_idx = begin - 1
        end_idx = start_idx + count
        if start_idx >= len(pairs):
            return []
        return [doc for doc, _ in pairs[start_idx:end_idx]]

    def search(self, content: str, n: int) -> list[str]:
        if n <= 0:
            return []

        query_embedding = self._embed_texts([content])[0]
        result = self.collection.query(
            query_embeddings=query_embedding[np.newaxis, :],
            n_results=n,
            include=["documents"],
        )
        documents_nested: list[list[str]] = result.get("documents", []) or []
        if not documents_nested:
            return []
        return [document for document in documents_nested[0] if document]

    def add(self, contents: list[str]) -> None:
        normalized_contents = [content.strip() for content in contents if content and content.strip()]
        if not normalized_contents:
            return

        added_memories_buffer: list[str] = []
        # Batch-embed incoming contents once to reduce per-item embedding overhead.
        content_embeddings = self._embed_texts(normalized_contents)

        for content, content_embedding in zip(normalized_contents, content_embeddings):
            related = self._find_related(content_embedding)

            if related:
                related_ids = [item["id"] for item in related]
                related_documents = [item["document"] for item in related]
                new_memories = self._synthesize_memories(related_documents, content)

                upsert_embeddings = (
                    content_embedding[np.newaxis, :]
                    if self._can_reuse_query_embedding(content, new_memories)
                    else None
                )
                self._upsert_memories(new_memories, embeddings=upsert_embeddings)
                self.collection.delete(ids=related_ids)
                added_memories_buffer.extend(new_memories)
            else:
                allow_multiple = self._should_allow_multiple_no_related(
                    content=content,
                    total_new_contents=len(normalized_contents),
                )
                new_memories = self._reconstruct_no_related_memories(content, allow_multiple=allow_multiple)
                upsert_embeddings = (
                    content_embedding[np.newaxis, :]
                    if self._can_reuse_query_embedding(content, new_memories)
                    else None
                )
                self._upsert_memories(new_memories, embeddings=upsert_embeddings)
                added_memories_buffer.extend(new_memories)

        if added_memories_buffer:
            self._update_core(added_memories_buffer)

    def _find_related(self, content_embedding: NDArray[np.float32]) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=content_embedding[np.newaxis, :],
            n_results=self.merge_top_k,
            include=["documents", "distances"],
        )

        ids_nested: list[list[str]] = result.get("ids", []) or []
        docs_nested: list[list[str]] = result.get("documents", []) or []
        distances_nested: list[list[float]] = result.get("distances", []) or []

        ids = ids_nested[0] if ids_nested else []
        docs = docs_nested[0] if docs_nested else []
        distances = distances_nested[0] if distances_nested else []

        related: list[dict[str, Any]] = []
        for idx, memory_id in enumerate(ids):
            document = docs[idx] if idx < len(docs) else ""
            distance = float(distances[idx]) if idx < len(distances) else float("inf")
            if not document:
                continue
            if distance <= self.merge_distance_cutoff:
                related.append(
                    {
                        "id": memory_id,
                        "document": document,
                        "distance": distance,
                    }
                )
        return related

    def _synthesize_memories(self, related_memories: list[str], new_content: str) -> list[str]:
        user_prompt = build_synthesize_user_prompt(related_memories=related_memories, new_content=new_content)
        response = self._generate_structured_with_retry(
            system_prompt=SYNTHESIZE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=MemorySynthesisResponse,
        )
        if response is None:
            return [new_content]
        memories = self._normalize_memories(response.update)
        return memories or [new_content]

    def _reconstruct_no_related_memories(self, content: str, allow_multiple: bool) -> list[str]:
        user_prompt = build_no_related_memory_user_prompt(content=content, allow_multiple=allow_multiple)
        response = self._generate_structured_with_retry(
            system_prompt=NO_RELATED_MEMORY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=NoRelatedMemoryResponse,
        )
        if response is None:
            return [content]
        memories = self._normalize_memories(response.update)
        return memories or [content]

    def _update_core(self, new_memories: list[str]) -> None:
        current_core = self.get_core()
        user_prompt = build_core_update_user_prompt(current_core=current_core, new_memories=new_memories)

        response = self._generate_structured_with_retry(
            system_prompt=CORE_UPDATE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=CoreUpdateResponse,
        )
        if response is None:
            return
        if response.should_update and self._is_valid_core_markdown(response.core_markdown):
            updated_core = response.core_markdown.strip() + "\n"
            self.core_path.write_text(updated_core, encoding="utf-8")

    def _generate_structured_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> ModelT | None:
        for _ in range(3):
            try:
                raw = self.llm_client.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                )
                parsed = response_model.model_validate(raw)
                return parsed
            except Exception:
                continue
        return None

    def _normalize_memories(self, memories: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for memory in memories:
            candidate = memory.strip()
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    def _upsert_memories(self, memories: list[str], embeddings: NDArray[np.float32] | None = None) -> None:
        if not memories:
            return

        if embeddings is None:
            embeddings_to_store = self._embed_texts(memories)
        else:
            embeddings_to_store = self._coerce_embeddings(memories, embeddings)
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for _ in memories:
            memory_id = str(uuid4())
            timestamp, timestamp_ms = self._next_timestamp()
            ids.append(memory_id)
            metadatas.append({"timestamp": timestamp, "timestamp_ms": timestamp_ms})

        self.collection.upsert(
            ids=ids,
            documents=memories,
            embeddings=embeddings_to_store,
            metadatas=metadatas,
        )

    def _embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        raw_embeddings = self.embedding_client.embed(texts, self.output_dimensionality)
        return self._coerce_embeddings(texts, raw_embeddings)

    def _coerce_embeddings(
        self,
        texts: list[str],
        embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                "Embedding batch must be 2D with shape (num_texts, output_dimensionality), "
                f"got ndim={array.ndim}"
            )
        if array.shape[0] != len(texts):
            raise ValueError(
                "Embedding client returned wrong number of vectors: "
                f"expected {len(texts)}, got {array.shape[0]}"
            )
        if array.shape[1] != self.output_dimensionality:
            raise ValueError(
                "Embedding vector dimensionality mismatch: "
                f"expected {self.output_dimensionality}, got {array.shape[1]}"
            )
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array, dtype=np.float32)
        return array

    def _can_reuse_query_embedding(self, content: str, memories: list[str]) -> bool:
        return len(memories) == 1 and memories[0] == content

    def _should_allow_multiple_no_related(self, content: str, total_new_contents: int) -> bool:
        if total_new_contents > 1:
            return True
        if len(content) >= 220:
            return True
        if "\n" in content:
            return True
        if any(marker in content for marker in ("; ", " - ", "•", "1.", "2.", "3.")):
            return True
        sentences = [part.strip() for part in re.split(r"[.!?]\s+", content) if part.strip()]
        return len(sentences) >= 2

    def _next_timestamp(self) -> tuple[str, int]:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        timestamp = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return timestamp, timestamp_ms

    def _extract_timestamp_ms(self, metadata: dict[str, Any] | None) -> int:
        if not metadata:
            return 0
        raw = metadata.get("timestamp_ms", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def _is_valid_core_markdown(self, value: str) -> bool:
        if not value or not value.strip():
            return False
        required_headers = ["# SOUL", "# TOOLS", "# RULE", "# USER"]
        return all(header in value for header in required_headers)
