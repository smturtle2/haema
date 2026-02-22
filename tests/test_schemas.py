from __future__ import annotations

from haema.schemas import CoreUpdateResponse, MemoryReconstructionResponse, PreMemorySplitResponse


def _description(model: type[object], field_name: str) -> str:
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    field = properties.get(field_name, {})
    return str(field.get("description", ""))


def test_pre_memory_split_schema_description_mentions_quality_criteria() -> None:
    description = _description(PreMemorySplitResponse, "contents")

    assert "independent factual topic" in description
    assert "duplicates/low-signal filler" in description
    assert "unsupported inference" in description


def test_reconstruction_schema_descriptions_define_priority_and_coverage() -> None:
    memories_description = _description(MemoryReconstructionResponse, "memories")
    coverage_description = _description(MemoryReconstructionResponse, "coverage")

    assert "integrating related memories with new contents" in memories_description
    assert "New evidence should be prioritized" in memories_description
    assert "atomic, reusable, non-duplicated" in memories_description

    assert "important actionable facts" in coverage_description
    assert "major omission" in coverage_description
    assert "important facts are missing" in coverage_description


def test_core_update_schema_descriptions_define_conservative_policy() -> None:
    should_update_description = _description(CoreUpdateResponse, "should_update")
    core_markdown_description = _description(CoreUpdateResponse, "core_markdown")

    assert "Prefer false unless candidate memories are durable" in should_update_description
    assert "materially improve future behavior" in should_update_description

    assert "Each bullet should belong to one section only" in core_markdown_description
    assert "avoid session/transient noise" in core_markdown_description
    assert "soft budget of about 8 total bullets" in core_markdown_description
