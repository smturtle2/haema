from __future__ import annotations

from typing import Literal
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class PreMemorySplitResponse(BaseModel):
    contents: list[str] = Field(
        ...,
        description=(
            "Expanded pre-memory text units derived from a single raw add() string input. "
            "Split independent facts into separate items while preserving essential detail."
        ),
    )


class MemoryReconstructionResponse(BaseModel):
    memories: list[str] = Field(
        ...,
        description=(
            "Final reconstructed long-term memory entries to store after combining "
            "related memories and new contents. Any number of entries is allowed, and "
            "each entry may be short or long depending on information density."
        ),
    )
    coverage: Literal["complete", "incomplete"] = Field(
        ...,
        description=(
            "Whether reconstructed memories sufficiently cover the high-value facts from "
            "new contents. Use 'complete' when coverage is adequate, otherwise 'incomplete'."
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
    core_markdown: Optional[str] = Field(
        None,
        description=(
            "The full core memory markdown text when should_update is true. "
            "It must include all required sections: # SOUL, # TOOLS, # RULE, # USER. "
            "May be null when should_update is false."
        ),
    )

    @model_validator(mode="after")
    def validate_core_markdown_when_updating(self) -> "CoreUpdateResponse":
        if self.should_update and not self.core_markdown:
            raise ValueError("core_markdown is required when should_update is true")
        return self
