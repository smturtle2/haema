from __future__ import annotations

from pydantic import BaseModel, Field


class MemorySynthesisResponse(BaseModel):
    update: list[str] = Field(
        ...,
        description=(
            "Synthesize the provided related memories and new input into an arbitrary "
            "number of updated long-term memory entries while preserving key signal and "
            "removing redundancy. Each entry may be short or long as needed."
        ),
    )


class NoRelatedMemoryResponse(BaseModel):
    update: list[str] = Field(
        ...,
        description=(
            "Reconstruct the new content into one or more durable memory entries when no "
            "related memory exists. Prefer one memory, but output multiple if the input "
            "contains multiple independent high-value facts. Entries may be short or long "
            "depending on required detail."
        ),
    )


class SingleMemoryResponse(BaseModel):
    memory: str = Field(
        ...,
        description=(
            "A single reconstructed memory entry distilled from one new content input. "
            "It may be short or long depending on how much detail is necessary."
        ),
    )


class CoreUpdateResponse(BaseModel):
    should_update: bool = Field(
        ...,
        description=(
            "Whether the existing core memory should be updated based on new memories. "
            "Use true only when critical SOUL/TOOLS/RULE/USER changes are required."
        ),
    )
    core_markdown: str = Field(
        ...,
        description=(
            "The full core memory markdown text. It must include all required sections: "
            "# SOUL, # TOOLS, # RULE, # USER."
        ),
    )
