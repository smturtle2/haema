from __future__ import annotations

import inspect
from pathlib import Path

from haema import EmbeddingClient, LLMClient, Memory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert_doc_contains(target: object, required_terms: list[str]) -> None:
    doc = inspect.getdoc(target)
    assert doc is not None
    assert doc.strip()
    for term in required_terms:
        assert term in doc


def test_public_classes_have_non_empty_docstrings() -> None:
    for cls in (Memory, EmbeddingClient, LLMClient):
        doc = inspect.getdoc(cls)
        assert doc is not None
        assert doc.strip()


def test_memory_public_method_docstrings_cover_usage_contract() -> None:
    _assert_doc_contains(
        Memory.__init__,
        [
            "Args:",
            "Returns:",
            "Raises:",
            "path",
            "output_dimensionality",
            "embedding_client",
            "llm_client",
            "merge_top_k",
            "merge_distance_cutoff",
            "Example:",
        ],
    )
    _assert_doc_contains(Memory.get_core, ["Returns:", "Example:", "core.md"])
    _assert_doc_contains(Memory.get_latest, ["Args:", "Returns:", "Raises:", "Behavior:", "Example:"])
    _assert_doc_contains(Memory.search, ["Args:", "Returns:", "Behavior:", "Example:"])
    _assert_doc_contains(Memory.add, ["Args:", "Returns:", "Raises:", "Behavior:", "Example:"])


def test_embedding_client_docstrings_define_shape_and_dtype_contract() -> None:
    required = ["Args:", "Returns:", "shape", "dtype", "float32", "output_dimensionality"]
    _assert_doc_contains(EmbeddingClient.embed_query, required)
    _assert_doc_contains(EmbeddingClient.embed_document, required)


def test_llm_client_generate_structured_docstring_defines_parsing_contract() -> None:
    _assert_doc_contains(
        LLMClient.generate_structured,
        ["Args:", "Returns:", "response_model", "model_validate", "dict"],
    )


def test_readme_and_api_docs_keep_constructor_signature_in_sync() -> None:
    constructor = (
        "Memory(path, output_dimensionality, embedding_client, llm_client, "
        "merge_top_k=3, merge_distance_cutoff=0.25)"
    )
    public_methods = [
        "get_core() -> str",
        "get_latest(begin: int, count: int) -> list[str]",
        "search(content: str, n: int) -> list[str]",
        "add(contents: str | list[str]) -> None",
    ]

    for relpath in ("README.md", "README.ko.md", "docs/api.md"):
        text = (PROJECT_ROOT / relpath).read_text(encoding="utf-8")
        assert constructor in text
        for method in public_methods:
            assert method in text
