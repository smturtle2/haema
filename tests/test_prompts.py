from __future__ import annotations

from haema.prompts import (
    CORE_UPDATE_SYSTEM_PROMPT,
    MEMORY_RECONSTRUCTION_SYSTEM_PROMPT,
    PRE_MEMORY_SPLIT_SYSTEM_PROMPT,
    build_core_update_user_prompt,
    build_pre_memory_split_user_prompt,
    build_reconstruction_user_prompt,
)


def test_pre_memory_split_user_prompt_has_tagged_input_boundary() -> None:
    prompt = build_pre_memory_split_user_prompt("alpha fact")

    assert "<raw_input>" in prompt
    assert "</raw_input>" in prompt
    assert "- alpha fact" in prompt
    assert "Do not perform core selection or policy decisions in this task." in prompt
    assert "Use only content inside <raw_input> as evidence." in prompt


def test_reconstruction_user_prompt_has_tagged_blocks_and_task_boundary() -> None:
    prompt = build_reconstruction_user_prompt(
        related_memories=["old alpha"],
        new_contents=["new beta"],
    )

    assert "Related memories:" in prompt
    assert "<related_memories>" in prompt
    assert "</related_memories>" in prompt
    assert "New contents:" in prompt
    assert "<new_contents>" in prompt
    assert "</new_contents>" in prompt
    assert "- old alpha" in prompt
    assert "- new beta" in prompt
    assert "Produce memory entries only (not core markdown)." in prompt
    assert "Use only facts grounded in the tagged input blocks." in prompt


def test_core_update_user_prompt_has_tagged_blocks_and_budget_instruction() -> None:
    prompt = build_core_update_user_prompt(
        current_core="# SOUL\n- stable\n\n# TOOLS\n\n# RULE\n\n# USER\n",
        new_memories=["new durable fact"],
    )

    assert "<current_core_markdown>" in prompt
    assert "</current_core_markdown>" in prompt
    assert "<candidate_new_memories>" in prompt
    assert "</candidate_new_memories>" in prompt
    assert "soft budget of <= 8 total bullets" in prompt
    assert "coreworthiness gates" in prompt


def test_system_prompts_include_stage_responsibility_and_core_gate_policy() -> None:
    assert "Do not perform policy decisions or core selection in this task." in PRE_MEMORY_SPLIT_SYSTEM_PROMPT
    assert "Reconstruct the next durable long-term memory set from two inputs." in MEMORY_RECONSTRUCTION_SYSTEM_PROMPT
    assert "Coreworthiness gate (ALL required for inclusion):" in CORE_UPDATE_SYSTEM_PROMPT
    assert "Durability: likely to remain useful across future sessions." in CORE_UPDATE_SYSTEM_PROMPT
    assert "Every kept item must belong to exactly one section." in CORE_UPDATE_SYSTEM_PROMPT
    assert "soft budget of <= 8 total bullets." in CORE_UPDATE_SYSTEM_PROMPT
