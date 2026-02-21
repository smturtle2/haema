from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from haema.clients import EmbeddingClient, LLMClient
from haema.prompts import (
    CORE_TEMPLATE,
    CORE_UPDATE_SYSTEM_PROMPT,
    MEMORY_RECONSTRUCTION_SYSTEM_PROMPT,
    PRE_MEMORY_SPLIT_SYSTEM_PROMPT,
    build_core_update_user_prompt,
    build_pre_memory_split_user_prompt,
    build_reconstruction_refine_user_prompt,
    build_reconstruction_user_prompt,
)
from haema.schemas import CoreUpdateResponse, MemoryReconstructionResponse, PreMemorySplitResponse

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
    RETRYABLE_LLM_EXCEPTIONS = (ValidationError, ValueError, TypeError, ConnectionError, TimeoutError, OSError)

    def __init__(
        self,
        path: str | Path,
        output_dimensionality: int,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
        merge_top_k: int = 3,
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
        self._logger = logging.getLogger(__name__)

        client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self.latest_db_path = self.path / "latest.sqlite3"
        self._latest_conn = sqlite3.connect(str(self.latest_db_path))
        self._latest_conn.execute("PRAGMA journal_mode=WAL;")
        self._init_latest_store()
        self._bootstrap_latest_store_if_needed()

    def get_core(self) -> str:
        return self.core_path.read_text(encoding="utf-8")

    def get_latest(self, begin: int, count: int) -> list[str]:
        if begin < 1:
            raise ValueError("begin must be >= 1")
        if count <= 0:
            return []

        offset = begin - 1
        cursor = self._latest_conn.execute(
            """
            SELECT document
            FROM latest_memories
            WHERE deleted = 0
            ORDER BY timestamp_ms DESC
            LIMIT ? OFFSET ?
            """,
            (count, offset),
        )
        return [row[0] for row in cursor.fetchall()]

    def search(self, content: str, n: int) -> list[str]:
        if n <= 0:
            return []

        query_embedding = self._embed_query_texts([content])[0]
        result = self.collection.query(
            query_embeddings=query_embedding[np.newaxis, :],
            n_results=n,
            include=["documents"],
        )
        documents_nested: list[list[str]] = result.get("documents", []) or []
        if not documents_nested:
            return []
        return [document for document in documents_nested[0] if document]

    def add(self, contents: str | list[str]) -> None:
        normalized_contents = self._normalize_contents_input(contents)
        if not normalized_contents:
            return

        content_embeddings = self._embed_query_texts(normalized_contents)
        related_ids, related_documents = self._collect_related_memories(content_embeddings)
        new_memories = self._reconstruct_memories(related_documents, normalized_contents)

        self._upsert_memories(new_memories)

        if related_ids:
            self.collection.delete(ids=related_ids)
            self._latest_mark_deleted(related_ids)

        self._update_core(new_memories)

    def _normalize_contents_input(self, contents: str | list[str]) -> list[str]:
        if isinstance(contents, str):
            return self._expand_single_content_with_llm(contents)

        normalized: list[str] = []
        for content in contents:
            if not isinstance(content, str):
                raise TypeError("contents must be str or list[str]")
            candidate = content.strip()
            if not candidate:
                continue
            normalized.append(candidate)
        return normalized

    def _expand_single_content_with_llm(self, content: str) -> list[str]:
        stripped = content.strip()
        if not stripped:
            return []

        user_prompt = build_pre_memory_split_user_prompt(stripped)
        response = self._generate_structured_with_retry(
            system_prompt=PRE_MEMORY_SPLIT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=PreMemorySplitResponse,
        )
        if response is None:
            return [stripped]

        expanded = self._normalize_memories(response.contents)
        return expanded or [stripped]

    def _collect_related_memories(self, content_embeddings: NDArray[np.float32]) -> tuple[list[str], list[str]]:
        related_by_id: dict[str, str] = {}
        for content_embedding in content_embeddings:
            related = self._find_related(content_embedding)
            for item in related:
                memory_id = item["id"]
                if memory_id in related_by_id:
                    continue
                related_by_id[memory_id] = item["document"]
        return list(related_by_id.keys()), list(related_by_id.values())

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

    def _reconstruct_memories(self, related_memories: list[str], new_contents: list[str]) -> list[str]:
        user_prompt = build_reconstruction_user_prompt(
            related_memories=related_memories,
            new_contents=new_contents,
        )
        response = self._generate_structured_with_retry(
            system_prompt=MEMORY_RECONSTRUCTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=MemoryReconstructionResponse,
        )
        if response is not None:
            memories = self._normalize_memories(response.memories)
            if memories and response.coverage == "complete":
                return memories

        refine_prompt = build_reconstruction_refine_user_prompt(user_prompt)
        refined_response = self._generate_structured_with_retry(
            system_prompt=MEMORY_RECONSTRUCTION_SYSTEM_PROMPT,
            user_prompt=refine_prompt,
            response_model=MemoryReconstructionResponse,
        )
        if refined_response is not None:
            refined_memories = self._normalize_memories(refined_response.memories)
            if refined_memories and refined_response.coverage == "complete":
                return refined_memories

        return list(new_contents)

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
        if response.should_update and response.core_markdown and self._is_valid_core_markdown(response.core_markdown):
            updated_core = response.core_markdown.strip() + "\n"
            self.core_path.write_text(updated_core, encoding="utf-8")

    def _generate_structured_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> ModelT | None:
        last_retryable_error: Exception | None = None
        for attempt in range(3):
            try:
                raw = self.llm_client.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                )
                parsed = response_model.model_validate(raw)
                return parsed
            except self.RETRYABLE_LLM_EXCEPTIONS as exc:
                last_retryable_error = exc
                self._logger.debug(
                    "Structured generation retry %s/3 failed for model %s: %s",
                    attempt + 1,
                    response_model.__name__,
                    exc,
                )
                continue
        if last_retryable_error is not None:
            self._logger.warning(
                "Structured generation failed after retries for model %s: %s",
                response_model.__name__,
                last_retryable_error,
            )
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

    def _upsert_memories(self, memories: list[str]) -> None:
        if not memories:
            return

        embeddings_to_store = self._embed_document_texts(memories)
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
        self._latest_upsert(ids, memories, metadatas)

    def _embed_query_texts(self, texts: list[str]) -> NDArray[np.float32]:
        raw_embeddings = self.embedding_client.embed_query(texts, self.output_dimensionality)
        return self._coerce_embeddings(texts, raw_embeddings)

    def _embed_document_texts(self, texts: list[str]) -> NDArray[np.float32]:
        raw_embeddings = self.embedding_client.embed_document(texts, self.output_dimensionality)
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

    def _next_timestamp(self) -> tuple[str, int]:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        timestamp = self._timestamp_from_ms(timestamp_ms)
        return timestamp, timestamp_ms

    def _extract_timestamp_ms(self, metadata: dict[str, Any] | None) -> int:
        if not metadata:
            return 0
        raw = metadata.get("timestamp_ms", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def _timestamp_from_ms(self, timestamp_ms: int) -> str:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _extract_timestamp(self, metadata: dict[str, Any] | None, timestamp_ms: int) -> str:
        if metadata:
            raw = metadata.get("timestamp")
            if isinstance(raw, str) and raw.strip():
                return raw
        return self._timestamp_from_ms(timestamp_ms)

    def _init_latest_store(self) -> None:
        self._latest_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS latest_memories (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._latest_conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_latest_memories_deleted_ts
            ON latest_memories(deleted, timestamp_ms DESC)
            """
        )
        self._latest_conn.commit()

    def _bootstrap_latest_store_if_needed(self) -> None:
        existing = self._latest_conn.execute("SELECT COUNT(1) FROM latest_memories").fetchone()
        if existing and int(existing[0]) > 0:
            return

        result = self.collection.get(include=["documents", "metadatas"])
        ids: list[str] = result.get("ids", []) or []
        documents: list[str] = result.get("documents", []) or []
        metadatas: list[dict[str, Any] | None] = result.get("metadatas", []) or []

        rows: list[tuple[str, str, str, int, int]] = []
        for idx, memory_id in enumerate(ids):
            document = documents[idx] if idx < len(documents) else ""
            if not document:
                continue
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            timestamp_ms = self._extract_timestamp_ms(metadata)
            timestamp = self._extract_timestamp(metadata, timestamp_ms)
            rows.append((memory_id, document, timestamp, timestamp_ms, 0))

        if rows:
            self._latest_conn.executemany(
                """
                INSERT OR REPLACE INTO latest_memories(id, document, timestamp, timestamp_ms, deleted)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._latest_conn.commit()

    def _latest_upsert(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        rows: list[tuple[str, str, str, int, int]] = []
        for idx, memory_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            timestamp_ms = self._extract_timestamp_ms(metadata)
            timestamp = self._extract_timestamp(metadata, timestamp_ms)
            rows.append((memory_id, documents[idx], timestamp, timestamp_ms, 0))

        if not rows:
            return

        self._latest_conn.executemany(
            """
            INSERT OR REPLACE INTO latest_memories(id, document, timestamp, timestamp_ms, deleted)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._latest_conn.commit()

    def _latest_mark_deleted(self, ids: list[str]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._latest_conn.execute(
            f"UPDATE latest_memories SET deleted = 1 WHERE id IN ({placeholders})",
            ids,
        )
        self._latest_conn.commit()

    def _is_valid_core_markdown(self, value: str) -> bool:
        if not value or not value.strip():
            return False
        required_headers = ["# SOUL", "# TOOLS", "# RULE", "# USER"]
        return all(header in value for header in required_headers)
