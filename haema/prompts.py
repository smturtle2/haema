from __future__ import annotations

CORE_TEMPLATE = """# SOUL

# TOOLS

# RULE

# USER
"""


PRE_MEMORY_SPLIT_SYSTEM_PROMPT = """You are a pre-memory expansion engine.
Task:
- Convert one raw input text into reusable pre-memory units.

Input:
- The source text is inside <raw_input> ... </raw_input>.
- Use only information from that tagged block.

Transformation rules:
1) Split independent facts/topics into separate entries when helpful.
2) Keep each entry atomic, self-contained, and retrieval-friendly.
3) Preserve important factual qualifiers needed for correct future reuse.
4) Remove duplication, filler, social niceties, and low-signal chatter.
5) Do not infer or invent unsupported facts.
6) Do not perform policy decisions or core selection in this task.

Output contract:
1) Output must match schema:
   - contents: list[str]
2) Returning one item is valid when the input is one atomic fact.

Return only structured output matching the schema."""


MEMORY_RECONSTRUCTION_SYSTEM_PROMPT = """You are a long-term memory reconstruction engine.
Task:
- Reconstruct the next durable long-term memory set from two inputs.

Inputs:
- Existing related memories are inside <related_memories> ... </related_memories>.
- New evidence is inside <new_contents> ... </new_contents>.
- Use only information from these tagged blocks.

Reconstruction logic:
1) Treat <new_contents> as higher-priority evidence by default.
2) Merge still-useful compatible facts from <related_memories>.
3) If conflict exists, prefer newer evidence unless coexistence is clearly valid.
4) Split independent facts into separate entries and merge duplicates.
5) Remove stale, redundant, or low-signal details.
6) Keep enough context for future retrieval/reasoning without bloating.

Output contract:
1) Output must match schema:
   - memories: list[str]
   - coverage: "complete" | "incomplete"
2) Each memory must be atomic, reusable, non-duplicated, and grounded in inputs.
3) Do not invent unsupported facts.
4) Output size may be smaller, equal, or larger than input size.
5) Coverage policy:
   - "complete": important actionable facts in <new_contents> are preserved.
   - "incomplete": important facts from <new_contents> are missing or over-compressed.

Return only structured output matching the schema."""


# Backward-compatible alias for older imports.
SYNTHESIZE_SYSTEM_PROMPT = MEMORY_RECONSTRUCTION_SYSTEM_PROMPT


CORE_UPDATE_SYSTEM_PROMPT = """You maintain an agent core memory markdown.
Task:
- Maintain a compact core markdown that contains only durable high-impact information.
- Be conservative. If uncertain, do not update.

Inputs:
- Current core is inside <current_core_markdown> ... </current_core_markdown>.
- Candidate new memories are inside <candidate_new_memories> ... </candidate_new_memories>.
- Use only information from these tagged blocks.

Coreworthiness gate (ALL required for inclusion):
1) Durability: likely to remain useful across future sessions.
2) Behavioral impact: materially affects future responses/decisions.
3) Confidence: strongly supported and non-speculative.
If any gate fails, reject that item from core.

Exclusion rules:
- Reject session-only intent, one-off tasks, temporary requests, and ephemeral context.
- Reject transient logs/outputs/incidents/outages.
- Reject speculative, low-confidence, or weakly supported statements.
- Reject details better suited for non-core memory layers.

Section routing (single ownership):
- Every kept item must belong to exactly one section.
- If placement is ambiguous across sections, reject that item.

[SOUL]
Durable identity, mission, communication philosophy.

[TOOLS]
Durable tool capability boundaries and tool-usage posture.

[RULE]
Non-negotiable constraints, mandatory checks, priority rules.

[USER]
Stable user profile and durable user preferences.

Merge and pruning policy:
1) Prefer merge/compress over append.
2) Resolve conflicts in favor of newer evidence unless coexistence is clearly valid.
3) Keep high signal density with a soft budget of <= 8 total bullets.
4) If over budget, keep the most durable/high-impact/high-confidence items.
5) Replace weaker outdated items when stronger new evidence exists.

Update decision:
1) should_update=true only when final core is meaningfully improved.
2) If no meaningful improvement, set should_update=false.

Output contract:
1) Output must support:
   - should_update: boolean
   - core_markdown: full markdown string or null
2) If should_update=false, core_markdown must be null.
3) If should_update=true, core_markdown must contain exactly these top-level headers:
   - # SOUL
   - # TOOLS
   - # RULE
   - # USER
4) Use plain markdown; concise bullet points are preferred.

Return only structured output matching the schema."""


def build_reconstruction_user_prompt(related_memories: list[str], new_contents: list[str]) -> str:
    related_block = "\n".join(f"- {memory}" for memory in related_memories) or "- (none)"
    new_block = "\n".join(f"- {content}" for content in new_contents) or "- (none)"
    return (
        "Related memories:\n"
        "<related_memories>\n"
        f"{related_block}\n"
        "</related_memories>\n\n"
        "New contents:\n"
        "<new_contents>\n"
        f"{new_block}\n"
        "</new_contents>\n\n"
        "Task: reconstruct all inputs into the next long-term memory set.\n"
        "Requirements:\n"
        "- Preserve critical information from new contents.\n"
        "- Remove duplication and stale/noisy details.\n"
        "- Give higher priority to NEW contents over older related memories.\n"
        "- If there is contradiction, prefer NEW contents unless coexistence is clearly valid.\n"
        "- Split independent facts into separate memories.\n"
        "- Keep necessary detail (do not force brevity).\n"
        "- Produce memory entries only (not core markdown).\n"
        "- Do not invent unsupported facts.\n"
        "- Use only facts grounded in the tagged input blocks.\n"
        "- Return only schema fields."
    )


def build_reconstruction_refine_user_prompt(base_prompt: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "Quality correction:\n"
        "- Previous output was incomplete, missing key facts, or over-compressed.\n"
        "- Re-check all major actionable facts from <new_contents> and restore omissions.\n"
        "- Split independent facts more aggressively when they are currently merged.\n"
        "- Keep coverage='complete' only when important facts are not omitted."
    )


def build_pre_memory_split_user_prompt(content: str) -> str:
    return (
        "Raw input text:\n"
        "<raw_input>\n"
        f"- {content}\n"
        "</raw_input>\n\n"
        "Task: expand this single input into multiple pre-memory text units when useful.\n"
        "Requirements:\n"
        "- Split independent facts/topics into reusable units.\n"
        "- Preserve key factual detail needed for later retrieval.\n"
        "- Remove duplicates, filler, and low-signal conversational residue.\n"
        "- Do not infer unsupported facts.\n"
        "- Do not perform core selection or policy decisions in this task.\n"
        "- Use only content inside <raw_input> as evidence.\n"
        "- Return schema fields only."
    )


def build_core_update_user_prompt(current_core: str, new_memories: list[str]) -> str:
    new_memories_block = "\n".join(f"- {memory}" for memory in new_memories) or "- (none)"
    return (
        "Current core markdown:\n"
        "<current_core_markdown>\n"
        f"{current_core}\n"
        "</current_core_markdown>\n\n"
        "Candidate new memories:\n"
        "<candidate_new_memories>\n"
        f"{new_memories_block}\n"
        "</candidate_new_memories>\n\n"
        "Task:\n"
        "1) Decide whether core update is necessary under strict conservative policy and coreworthiness gates.\n"
        "2) If needed, return full updated markdown with # SOUL, # TOOLS, # RULE, # USER sections.\n"
        "3) If current core conflicts with candidate new memories, prioritize the newer evidence by default.\n"
        "4) Keep only durable, high-impact, high-confidence, cross-session information.\n"
        "5) Remove duplicates, avoid temporary/session-specific details, and keep a soft budget of <= 8 total bullets.\n"
        "6) Use only facts grounded in the tagged input blocks."
    )
