from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import haema.memory as memory_module
from haema.clients import EmbeddingClient, LLMClient
from haema.memory import Memory
from haema.schemas import CoreUpdateResponse, MemorySynthesisResponse, NoRelatedMemoryResponse


class FakeCollection:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: Any,
        metadatas: list[dict[str, Any]],
    ) -> None:
        for idx, memory_id in enumerate(ids):
            self._rows[memory_id] = {
                "document": documents[idx],
                "embedding": np.asarray(embeddings[idx], dtype=np.float32),
                "metadata": metadatas[idx],
            }

    def delete(self, ids: list[str]) -> None:
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
    def embed(self, texts: list[str], output_dimensionality: int) -> np.ndarray:
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


class CountingEmbeddingClient(EmbeddingClient):
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str], output_dimensionality: int) -> np.ndarray:
        self.calls += 1
        return KeywordEmbeddingClient().embed(texts, output_dimensionality)


class StubLLMClient(LLMClient):
    def __init__(self) -> None:
        self.failures_left: dict[type[Any], int] = {}
        self.synthesis_output = ["merged alpha A", "merged alpha B"]
        self.no_related_output: list[str] | None = None
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

        if response_model is NoRelatedMemoryResponse:
            if self.no_related_output is not None:
                return {"update": self.no_related_output}
            return {"update": [f"single::{_extract_first_bullet(user_prompt)}"]}
        if response_model is MemorySynthesisResponse:
            return {"update": self.synthesis_output}
        if response_model is CoreUpdateResponse:
            self.core_call_count += 1
            return {"should_update": self.core_should_update, "core_markdown": self.core_markdown}
        raise AssertionError("unexpected response_model")


class EchoLLMClient(LLMClient):
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
    ) -> dict[str, Any]:
        _ = system_prompt
        if response_model is NoRelatedMemoryResponse:
            return {"update": [_extract_first_bullet(user_prompt)]}
        if response_model is MemorySynthesisResponse:
            return {"update": [_extract_last_bullet(user_prompt)]}
        if response_model is CoreUpdateResponse:
            return {"should_update": False, "core_markdown": "# SOUL\n\n# TOOLS\n\n# RULE\n\n# USER\n"}
        raise AssertionError("unexpected response_model")


def _extract_first_bullet(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("- "):
            return line[2:].strip()
    return "unknown"


def _extract_last_bullet(prompt: str) -> str:
    bullets = [line[2:].strip() for line in prompt.splitlines() if line.startswith("- ")]
    return bullets[-1] if bullets else "unknown"


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


def _build_memory(tmp_path: Path, llm_client: StubLLMClient | None = None) -> tuple[Memory, StubLLMClient]:
    llm = llm_client or StubLLMClient()
    memory = Memory(
        path=tmp_path / "db",
        output_dimensionality=3,
        embedding_client=KeywordEmbeddingClient(),
        llm_client=llm,
    )
    return memory, llm


def test_initialization_creates_core_file(tmp_path: Path) -> None:
    memory, _ = _build_memory(tmp_path)
    core_path = tmp_path / "db" / "core.md"
    db_dir_path = tmp_path / "db" / "db"
    assert core_path.exists()
    assert db_dir_path.exists()
    assert db_dir_path.is_dir()
    core = memory.get_core()
    assert "# SOUL" in core
    assert "# TOOLS" in core
    assert "# RULE" in core
    assert "# USER" in core


def test_add_no_related_adds_single_memory_and_timestamp(tmp_path: Path) -> None:
    memory, _ = _build_memory(tmp_path)

    result = memory.add(["alpha first"])
    assert result is None

    records = memory.collection.get(include=["documents", "metadatas"])
    assert records["documents"] == ["single::alpha first"]
    metadata = records["metadatas"][0]
    assert metadata["timestamp"].endswith("Z")
    assert isinstance(metadata["timestamp_ms"], int)


def test_add_with_related_merges_and_deletes_related(tmp_path: Path) -> None:
    memory, llm = _build_memory(tmp_path)
    memory.add(["alpha seed"])

    llm.synthesis_output = ["merged alpha A", "merged alpha B"]
    memory.add(["alpha next"])

    records = memory.collection.get(include=["documents"])
    assert sorted(records["documents"]) == ["merged alpha A", "merged alpha B"]
    assert "single::alpha seed" not in records["documents"]


def test_add_fallback_when_llm_fails_three_times(tmp_path: Path) -> None:
    llm = StubLLMClient()
    llm.failures_left[NoRelatedMemoryResponse] = 3
    memory, _ = _build_memory(tmp_path, llm_client=llm)

    memory.add(["beta fallback"])

    records = memory.collection.get(include=["documents"])
    assert records["documents"] == ["beta fallback"]


def test_add_no_related_can_store_multiple_memories_when_needed(tmp_path: Path) -> None:
    llm = StubLLMClient()
    llm.no_related_output = ["user prefers terse answers", "user focuses on memory quality"]
    memory, _ = _build_memory(tmp_path, llm_client=llm)

    memory.add(["User prefers terse answers. User focuses on memory quality."])

    records = memory.collection.get(include=["documents"])
    assert sorted(records["documents"]) == sorted(llm.no_related_output)


def test_core_update_runs_once_per_add_call(tmp_path: Path) -> None:
    memory, llm = _build_memory(tmp_path)
    llm.core_markdown = (
        "# SOUL\nUpdated once.\n\n# TOOLS\nUpdated once.\n\n# RULE\nUpdated once.\n\n# USER\nUpdated once.\n"
    )

    memory.add(["alpha one", "beta two"])

    assert llm.core_call_count == 1
    assert "Updated once." in memory.get_core()


def test_get_latest_returns_descending_timestamp_order(tmp_path: Path) -> None:
    memory, _ = _build_memory(tmp_path)
    memory.add(["alpha one"])
    memory.add(["beta two"])
    memory.add(["gamma three"])

    latest = memory.get_latest(begin=1, count=2)
    assert latest == ["single::gamma three", "single::beta two"]
    assert memory.get_latest(begin=10, count=2) == []


def test_search_returns_top_n_documents(tmp_path: Path) -> None:
    memory, _ = _build_memory(tmp_path)
    memory.add(["alpha one"])
    memory.add(["beta two"])
    memory.add(["gamma three"])

    result = memory.search("alpha query", 2)
    assert len(result) == 2
    assert result[0] == "single::alpha one"


def test_public_api_return_types(tmp_path: Path) -> None:
    memory, _ = _build_memory(tmp_path)
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


def test_add_uses_batched_query_embedding_and_reuses_when_possible(tmp_path: Path) -> None:
    embedding_client = CountingEmbeddingClient()
    memory = Memory(
        path=tmp_path / "db",
        output_dimensionality=3,
        embedding_client=embedding_client,
        llm_client=EchoLLMClient(),
    )

    memory.add(["alpha one", "beta two", "gamma three"])

    # One embed call for all incoming contents, and upsert reuses per-content query embeddings.
    assert embedding_client.calls == 1
