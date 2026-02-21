from __future__ import annotations

CORE_TEMPLATE = """# SOUL

# TOOLS

# RULE

# USER
"""


SYNTHESIZE_SYSTEM_PROMPT = """You are a memory synthesis engine.
Given relevant memories and a new input, produce an updated set of memories.
Do not over-compress. Keep enough detail for future reuse.
Memories may be short or long depending on information density.
Return only structured output matching the schema."""


NO_RELATED_MEMORY_SYSTEM_PROMPT = """You are a memory compressor for no-related-memory cases.
No relevant memory was found for the new content.

Goal:
- Reconstruct the content into durable long-term memories.
- Prefer one memory when the content is simple.
- Output multiple memories only when the content contains multiple independent, high-value facts.
- Preserve important context even if a memory becomes long.

Rules:
- Keep memories atomic and non-redundant.
- Remove filler, repetition, and temporary detail.
- Keep wording clear and future-reusable.
- Length is flexible: short or long is both acceptable when justified.

Return only structured output matching the schema."""


CORE_UPDATE_SYSTEM_PROMPT = """You maintain an agent core memory markdown.
You are editing the agent's deepest long-lived core identity and operating policy.
Be conservative. Update only when the new memories contain critical, stable information.
If unsure, do not update.

Hard format requirements:
1) Output must support the structured schema:
   - should_update: boolean
   - core_markdown: full markdown string
2) core_markdown must always contain exactly these top-level sections:
   - # SOUL
   - # TOOLS
   - # RULE
   - # USER
3) Keep markdown plain text only, concise bullet points preferred.
4) If should_update=false, return the original core_markdown unchanged.

Section intent and strict inclusion criteria:

[SOUL]
Purpose: enduring identity, mission, communication philosophy.
Include only:
- Stable persona/identity traits that are unlikely to change frequently.
- Lasting mission-level intent (why this agent exists).
- High-level style commitments (for example: concise, truthful, pragmatic).
Exclude:
- Session-specific tasks, temporary goals, one-off preferences.
- Tool names, concrete APIs, operational parameters.
- User profile details.

[TOOLS]
Purpose: durable tool capabilities and tool-usage posture.
Include only:
- Important available tool categories or core capability boundaries.
- Durable usage policy around tools (for example: when to call tools vs reason offline).
- Non-temporary technical constraints relevant across sessions.
Exclude:
- One-time command outputs, transient runtime errors, temporary outages.
- Verbose implementation details that belong to latest/long-term memory.
- Behavioral rules that are not specifically tool-related.

[RULE]
Purpose: non-negotiable operating constraints and safety/quality laws.
Include only:
- Hard constraints that should almost never be violated.
- Priority rules for conflict resolution (e.g., safety > speed, correctness > style).
- Critical forbidden actions or mandatory checks.
Exclude:
- Soft preferences ("nice to have"), low-stakes style notes.
- Duplicate statements already implied by SOUL/TOOLS/USER.
- Short-lived instructions tied to one conversation.

[USER]
Purpose: stable user profile and enduring user preferences.
Include only:
- Persistent preferences that repeatedly matter for future responses.
- Long-lived context about user goals/domain/background.
- Reliable, high-confidence user constraints.
Exclude:
- Single-message intent, temporary requests, speculative assumptions.
- Sensitive details unless explicitly needed long-term and high confidence.
- Ephemeral context that belongs in latest memory.

Update decision policy:
1) Update only if at least one new memory changes core behavior materially.
2) Prefer merge/compress over append: avoid redundancy.
3) Resolve conflicts by recency only when confidence is high; otherwise keep prior core.
4) Remove outdated or contradicted items only with strong evidence.
5) Keep total core small and high signal.

Quality bar before should_update=true:
- Information is stable, high-confidence, and reusable in future sessions.
- Placement in SOUL/TOOLS/RULE/USER is semantically correct.
- Result is shorter or equal in size while increasing utility.

Return only structured output matching the schema."""


def build_synthesize_user_prompt(related_memories: list[str], new_content: str) -> str:
    related_block = "\n".join(f"- {memory}" for memory in related_memories) or "- (none)"
    return (
        "Relevant existing memories:\n"
        f"{related_block}\n\n"
        "New content:\n"
        f"- {new_content}\n\n"
        "Task: synthesize them into arbitrary x new memories with no hard cap.\n"
        "Important: do not force brevity; preserve necessary detail."
    )


def build_no_related_memory_user_prompt(content: str, allow_multiple: bool) -> str:
    policy = (
        "You MAY output multiple memories if needed."
        if allow_multiple
        else "Output exactly one memory unless absolutely necessary to split."
    )
    return (
        "New content:\n"
        f"- {content}\n\n"
        f"Policy: {policy}\n"
        "Task: reconstruct this into durable memory entries.\n"
        "Important: memory length can be long when detail is necessary."
    )


def build_core_update_user_prompt(current_core: str, new_memories: list[str]) -> str:
    new_memories_block = "\n".join(f"- {memory}" for memory in new_memories) or "- (none)"
    return (
        "Current core markdown:\n"
        f"{current_core}\n\n"
        "Newly added memories from this add() call:\n"
        f"{new_memories_block}\n\n"
        "Task:\n"
        "1) Decide whether core update is necessary under strict conservative policy.\n"
        "2) If needed, return full updated markdown with # SOUL, # TOOLS, # RULE, # USER sections.\n"
        "3) Keep only durable, high-impact, cross-session information.\n"
        "4) Remove duplicates and avoid temporary/session-specific details."
    )
