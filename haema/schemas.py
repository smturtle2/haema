from __future__ import annotations

from typing import Literal
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class PreMemorySplitResponse(BaseModel):
    contents: list[str] = Field(
        ...,
        description=(
            "Decomposed pre-memory units from one raw input text. Each entry should represent "
            "an independent factual topic, remain self-contained, preserve essential detail for "
            "future retrieval, remove duplicates/low-signal filler, and avoid unsupported inference."
        ),
    )


class MemoryReconstructionResponse(BaseModel):
    memories: list[str] = Field(
        ...,
        description=(
            "Reconstructed long-term memory entries after integrating related memories with "
            "new contents. New evidence should be prioritized on conflict. Entries should be "
            "atomic, reusable, non-duplicated, and fully grounded in provided inputs."
        ),
    )
    coverage: Literal["complete", "incomplete"] = Field(
        ...,
        description=(
            "Coverage assessment for high-value facts in new contents. Use 'complete' only "
            "when important actionable facts are preserved without major omission or harmful "
            "over-compression. Use 'incomplete' when important facts are missing."
        ),
    )


class CoreUpdateResponse(BaseModel):
    should_update: bool = Field(
        ...,
        description=(
            "Whether core markdown should be updated under a conservative policy. Prefer false "
            "unless candidate memories are durable, high-confidence, and materially improve future "
            "behavior in SOUL/TOOLS/RULE/USER."
        ),
    )
    core_markdown: Optional[str] = Field(
        None,
        description=(
            "The full core memory markdown text when should_update is true. "
            "It must include all required sections: # SOUL, # TOOLS, # RULE, # USER. "
            "Each bullet should belong to one section only, keep only durable high-signal "
            "information, avoid session/transient noise, and aim for a compact soft budget "
            "of about 8 total bullets. May be null when should_update is false."
        ),
    )

    @model_validator(mode="after")
    def validate_core_markdown_when_updating(self) -> "CoreUpdateResponse":
        if self.should_update and not self.core_markdown:
            raise ValueError("core_markdown is required when should_update is true")
        return self
