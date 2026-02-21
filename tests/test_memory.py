from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import haema.memory as memory_module
from haema.clients import EmbeddingClient, LLMClient
from haema.memory import Memory
from haema.schemas import CoreUpdateResponse, MemoryReconstructionResponse, PreMemorySplitResponse


class FakeCollection:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self.events: list[tuple[str, list[str]]] = []

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: Any,
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.events.append(("upsert", list(ids)))
        for idx, memory_id in enumerate(ids):
            self._rows[memory_id] = {
                "document": documents[idx],
                "embedding": np.asarray(embeddings[idx], dtype=np.float32),
                "metadata": metadatas[idx],
            }

    def delete(self, ids: list[str]) -> None:
        self.events.append(("delete", list(ids)))
        for memory_id in ids:
            self._rows.pop(memory_id, None)

    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        include = include or []
        ids = list(self._rows.keys())
        result: dict[str, Any] = {"ids": ids}
        if "documents" in include:
            result["documents"] = [self._rows[memory_id]["document"] for memory_id in ids]
        if "metadatas" in include:
            result["metadatas"] = [self._rows[memory_id]["metadata"] for memory_id in ids]
        return result

    def query(
        self,
        query_embeddings: Any,
        n_results: int,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        include = include or []
        query = np.asarray(query_embeddings[0], dtype=np.float32)
        ranked: list[tuple[str, float]] = []
        for memory_id, row in self._rows.items():
            distance = _cosine_distance(query, row["embedding"])
            ranked.append((memory_id, distance))
        ranked.sort(key=lambda item: item[1])
        ranked = ranked[:n_results]

        ids = [memory_id for memory_id, _ in ranked]
        result: dict[str, Any] = {"ids": [ids]}
        if "documents" in include:
            result["documents"] = [[self._rows[memory_id]["document"] for memory_id in ids]]
        if "metadatas" in include:
            result["metadatas"] = [[self._rows[memory_id]["metadata"] for memory_id in ids]]
        if "distances" in include:
            result["distances"] = [[distance for _, distance in ranked]]
        return result


class FakePersistentClient:
    def __init__(self, path: str) -> None:
        self.path = path
        self._collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> FakeCollection:
        _ = metadata
        if name not in self._collections:
            self._collections[name] = FakeCollection()
        return self._collections[name]


class FakeChromaModule:
    PersistentClient = FakePersistentClient


class KeywordEmbeddingClient(EmbeddingClient):
    def __init__(self) -> None:
        self.query_calls = 0
        self.document_calls = 0
        self.last_query_texts: list[str] = []

    def embed_query(self, texts: list[str], output_dimensionality: int) -> np.ndarray:
        self.query_calls += 1
        self.last_query_texts = list(texts)
        return self._embed(texts, output_dimensionality)

    def embed_document(self, texts: list[str], output_dimensionality: int) -> np.ndarray:
        self.document_calls += 1
        return self._embed(texts, output_dimensionality)

    def _embed(self, texts: list[str], output_dimensionality: int) -> np.ndarray:
        if output_dimensionality != 3:
            raise ValueError("this test embedding client expects output_dimensionality=3")
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            elif "gamma" in lowered:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([1.0, 1.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class StubLLMClient(LLMClient):
    def __init__(self) -> None:
        self.failures_left: dict[type[Any], int] = {}
        self.split_responses: list[dict[str, Any]] = []
        self.split_call_count = 0
        self.reconstruction_responses: list[dict[str, Any]] = []
        self.reconstruction_call_count = 0
        self.reconstruction_user_prompts: list[str] = []
        self.core_call_count = 0
        self.core_markdown = (
            "# SOUL\nAgent identity.\n\n# TOOLS\nTool policy.\n\n# RULE\nCritical rule.\n\n# USER\nUser profile.\n"
        )
        self.core_should_update = True

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
    ) -> dict[str, Any]:
        _ = system_prompt
        if self.failures_left.get(response_model, 0) > 0:
            self.failures_left[response_model] -= 1
            raise ValueError("forced structured parsing failure")

        if response_model is PreMemorySplitResponse:
            self.split_call_count += 1
            if self.split_responses:
                return self.split_responses.pop(0)
            return {"contents": [_extract_first_raw_input(user_prompt)]}
        if response_model is MemoryReconstructionResponse:
            self.reconstruction_call_count += 1
            self.reconstruction_user_prompts.append(user_prompt)
            if self.reconstruction_responses:
                return self.reconstruction_responses.pop(0)
            return {"memories": [f"single::{_extract_first_new_content(user_prompt)}"], "coverage": "complete"}
        if response_model is CoreUpdateResponse:
            self.core_call_count += 1
            return {"should_update": self.core_should_update, "core_markdown": self.core_markdown}
        raise AssertionError("unexpected response_model")


class BrokenLLMClient(LLMClient):
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
    ) -> dict[str, Any]:
        _ = system_prompt
        _ = user_prompt
        _ = response_model
        raise AttributeError("broken llm client implementation")


def _extract_first_new_content(prompt: str) -> str:
    marker = "New contents:"
    marker_index = prompt.find(marker)
    if marker_index < 0:
        return "unknown"

    block = prompt[marker_index + len(marker) :]
    for line in block.splitlines():
        if line.startswith("Task:"):
            break
        if line.startswith("- "):
            return line[2:].strip()
    return "unknown"


def _extract_first_raw_input(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("- "):
            return line[2:].strip()
    return "unknown"


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cosine_similarity = dot / (norm_a * norm_b)
    return 1.0 - cosine_similarity


@pytest.fixture(autouse=True)
def patch_chromadb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_module, "chromadb", FakeChromaModule)


def _build_memory(
    tmp_path: Path,
    llm_client: StubLLMClient | None = None,
    embedding_client: KeywordEmbeddingClient | None = None,
) -> tuple[Memory, StubLLMClient, KeywordEmbeddingClient]:
    llm = llm_client or StubLLMClient()
    embedding = embedding_client or KeywordEmbeddingClient()
    memory = Memory(
        path=tmp_path / "db",
        output_dimensionality=3,
        embedding_client=embedding,
        llm_client=llm,
    )
    return memory, llm, embedding


def test_initialization_creates_core_file(tmp_path: Path) -> None:
    memory, _, _ = _build_memory(tmp_path)
    core_path = tmp_path / "db" / "core.md"
    db_dir_path = tmp_path / "db" / "db"
    latest_db_path = tmp_path / "db" / "latest.sqlite3"
    assert core_path.exists()
    assert db_dir_path.exists()
    assert db_dir_path.is_dir()
    assert latest_db_path.exists()
    assert memory.merge_top_k == 3
    core = memory.get_core()
    assert "# SOUL" in core
    assert "# TOOLS" in core
    assert "# RULE" in core
    assert "# USER" in core


def test_add_no_related_can_store_multiple_memories_and_timestamp(tmp_path: Path) -> None:
    llm = StubLLMClient()
    llm.reconstruction_responses = [
        {
            "memories": ["user prefers terse answers", "user focuses on memory quality"],
            "coverage": "complete",
        }
    ]
    memory, _, _ = _build_memory(tmp_path, llm_client=llm)

    result = memory.add(["User prefers terse answers. User focuses on memory quality."])
    assert result is None

    records = memory.collection.get(include=["documents", "metadatas"])
    assert sorted(records["documents"]) == [
        "user focuses on memory quality",
        "user prefers terse answers",
    ]
    metadata = records["metadatas"][0]
    assert metadata["timestamp"].endswith("Z")
    assert isinstance(metadata["timestamp_ms"], int)


def test_add_with_related_uses_union_dedupe_and_replaces_related(tmp_path: Path) -> None:
    memory, llm, _ = _build_memory(tmp_path)
    memory.add(["alpha seed"])

    llm.reconstruction_responses = [
        {"memories": ["merged alpha A", "merged alpha B"], "coverage": "complete"}
    ]
    memory.add(["alpha next one", "alpha next two"])

    records = memory.collection.get(include=["documents"])
    assert sorted(records["documents"]) == ["merged alpha A", "merged alpha B"]
    assert "single::alpha seed" not in records["documents"]

    delete_events = [event for event in memory.collection.events if event[0] == "delete"]
    assert len(delete_events) == 1
    assert len(delete_events[0][1]) == 1


def test_add_mixed_inputs_include_all_new_contents_in_single_prompt(tmp_path: Path) -> None:
    memory, llm, _ = _build_memory(tmp_path)
    memory.add(["alpha seed"])

    llm.reconstruction_responses = [
        {"memories": ["merged from mixed input"], "coverage": "complete"}
    ]
    memory.add(["alpha update", "beta separate"])

    assert llm.reconstruction_call_count >= 2
    final_prompt = llm.reconstruction_user_prompts[-1]
    assert "- alpha update" in final_prompt
    assert "- beta separate" in final_prompt


def test_add_retries_once_when_coverage_incomplete(tmp_path: Path) -> None:
    llm = StubLLMClient()
    llm.reconstruction_responses = [
        {"memories": ["collapsed memory"], "coverage": "incomplete"},
        {"memories": ["expanded memory A", "expanded memory B"], "coverage": "complete"},
    ]
    memory, _, _ = _build_memory(tmp_path, llm_client=llm)

    memory.add(["alpha one", "beta two"])

    records = memory.collection.get(include=["documents"])
    assert sorted(records["documents"]) == ["expanded memory A", "expanded memory B"]
    assert llm.reconstruction_call_count == 2


def test_add_fallback_when_reconstruction_fails(tmp_path: Path) -> None:
    llm = StubLLMClient()
    llm.failures_left[MemoryReconstructionResponse] = 6
    memory, _, _ = _build_memory(tmp_path, llm_client=llm)

    memory.add(["beta fallback"])

    records = memory.collection.get(include=["documents"])
    assert records["documents"] == ["beta fallback"]


def test_upsert_happens_before_delete_for_related_replacement(tmp_path: Path) -> None:
    memory, llm, _ = _build_memory(tmp_path)
    memory.add(["alpha seed"])

    llm.reconstruction_responses = [
        {"memories": ["merged alpha A"], "coverage": "complete"}
    ]
    memory.add(["alpha next"])

    assert memory.collection.events[-2][0] == "upsert"
    assert memory.collection.events[-1][0] == "delete"


def test_core_update_runs_once_per_add_call(tmp_path: Path) -> None:
    memory, llm, _ = _build_memory(tmp_path)
    llm.core_markdown = (
        "# SOUL\nUpdated once.\n\n# TOOLS\nUpdated once.\n\n# RULE\nUpdated once.\n\n# USER\nUpdated once.\n"
    )

    memory.add(["alpha one", "beta two"])

    assert llm.core_call_count == 1
    assert "Updated once." in memory.get_core()


def test_get_latest_returns_descending_timestamp_order(tmp_path: Path) -> None:
    memory, _, _ = _build_memory(tmp_path)
    memory.add(["alpha one"])
    memory.add(["beta two"])
    memory.add(["gamma three"])

    latest = memory.get_latest(begin=1, count=2)
    assert latest == ["single::gamma three", "single::beta two"]
    assert memory.get_latest(begin=10, count=2) == []


def test_get_latest_raises_for_invalid_begin(tmp_path: Path) -> None:
    memory, _, _ = _build_memory(tmp_path)
    with pytest.raises(ValueError):
        memory.get_latest(begin=0, count=1)


def test_get_latest_excludes_deleted_related_memories(tmp_path: Path) -> None:
    memory, llm, _ = _build_memory(tmp_path)
    memory.add(["alpha seed"])
    llm.reconstruction_responses = [
        {"memories": ["merged alpha A"], "coverage": "complete"}
    ]
    memory.add(["alpha next"])

    latest = memory.get_latest(begin=1, count=10)
    assert latest == ["merged alpha A"]
    assert "single::alpha seed" not in latest


def test_search_returns_top_n_documents_and_uses_query_embedding(tmp_path: Path) -> None:
    memory, _, embedding = _build_memory(tmp_path)
    memory.add(["alpha one"])
    memory.add(["beta two"])
    memory.add(["gamma three"])

    query_calls_before = embedding.query_calls
    result = memory.search("alpha query", 2)
    assert len(result) == 2
    assert result[0] == "single::alpha one"
    assert embedding.query_calls == query_calls_before + 1


def test_add_uses_query_and_document_embeddings_once_each(tmp_path: Path) -> None:
    embedding = KeywordEmbeddingClient()
    memory = Memory(
        path=tmp_path / "db",
        output_dimensionality=3,
        embedding_client=embedding,
        llm_client=StubLLMClient(),
    )

    memory.add(["alpha one", "beta two", "gamma three"])

    assert embedding.query_calls == 1
    assert embedding.document_calls == 1


def test_add_accepts_str_and_splits_into_multiple_contents(tmp_path: Path) -> None:
    llm = StubLLMClient()
    llm.split_responses = [
        {"contents": ["alpha first.", "beta second.", "gamma third."]}
    ]
    llm.reconstruction_responses = [
        {"memories": ["m1", "m2", "m3"], "coverage": "complete"}
    ]
    embedding = KeywordEmbeddingClient()
    memory = Memory(
        path=tmp_path / "db",
        output_dimensionality=3,
        embedding_client=embedding,
        llm_client=llm,
    )

    memory.add("alpha first.\n- beta second.\n3) gamma third.")

    assert embedding.last_query_texts == ["alpha first.", "beta second.", "gamma third."]
    assert llm.split_call_count == 1
    final_prompt = llm.reconstruction_user_prompts[-1]
    assert "- alpha first." in final_prompt
    assert "- beta second." in final_prompt
    assert "- gamma third." in final_prompt


def test_add_str_split_falls_back_to_original_input_on_failure(tmp_path: Path) -> None:
    llm = StubLLMClient()
    llm.failures_left[PreMemorySplitResponse] = 3
    embedding = KeywordEmbeddingClient()
    memory = Memory(
        path=tmp_path / "db",
        output_dimensionality=3,
        embedding_client=embedding,
        llm_client=llm,
    )

    memory.add("alpha one. beta two.")

    assert embedding.last_query_texts == ["alpha one. beta two."]


def test_memory_reconstruction_response_schema() -> None:
    parsed = MemoryReconstructionResponse.model_validate(
        {"memories": ["m1", "m2"], "coverage": "complete"}
    )
    assert parsed.memories == ["m1", "m2"]
    assert parsed.coverage == "complete"
    with pytest.raises(ValidationError):
        MemoryReconstructionResponse.model_validate({"memories": ["m1"], "coverage": "partial"})


def test_core_update_response_requires_markdown_when_should_update_true() -> None:
    with pytest.raises(ValidationError):
        CoreUpdateResponse.model_validate({"should_update": True, "core_markdown": None})


def test_public_api_return_types(tmp_path: Path) -> None:
    memory, _, _ = _build_memory(tmp_path)
    add_result = memory.add(["alpha one"])
    core_result = memory.get_core()
    latest_result = memory.get_latest(begin=1, count=10)
    search_result = memory.search("alpha query", 1)

    assert add_result is None
    assert isinstance(core_result, str)
    assert isinstance(latest_result, list)
    assert all(isinstance(item, str) for item in latest_result)
    assert isinstance(search_result, list)
    assert all(isinstance(item, str) for item in search_result)


def test_non_retryable_llm_error_propagates(tmp_path: Path) -> None:
    memory = Memory(
        path=tmp_path / "db",
        output_dimensionality=3,
        embedding_client=KeywordEmbeddingClient(),
        llm_client=BrokenLLMClient(),
    )
    with pytest.raises(AttributeError):
        memory.add(["alpha one"])
