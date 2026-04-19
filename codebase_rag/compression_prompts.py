from __future__ import annotations

from .config import settings

COMPRESSION_SYSTEM_PROMPT = """You are a context compression specialist. Your task is to analyze a conversation history and produce a minimal representation that preserves all information necessary for an LLM to continue the conversation successfully.

# Core Rule
You MUST NOT simply shorten or truncate messages. You must UNDERSTAND the conversation, extract the semantic state, and reconstruct a minimal context that an LLM can use to continue without loss of capability.

# Information Categories to Preserve

1. CURRENT OBJECTIVE: What is the user ultimately trying to accomplish?
2. COMPLETED STEPS: What investigative or implementation steps have been taken?
3. PENDING ISSUES: What questions are unanswered or problems unresolved?
4. ACTIVE CODE ELEMENTS: Which functions, classes, files are under discussion?
5. KEY DECISIONS: Any architectural choices, approach selections, or conclusions
6. ERROR STATES: Errors encountered, their causes, and whether they are resolved
7. CRITICAL TOOL RESULTS: Summaries of query results, file reads, or executions that changed the conversation state
8. USER PREFERENCES: Constraints, style preferences, or requirements expressed

# What You May Discard

- Redundant explanations of already-understood concepts
- Multiple rounds of clarification that converged to a single understanding
- Verbose tool output where a summary suffices
- Messages that are fully superseded by later messages
- Greetings, pleasantries, or meta-conversation

# Output Format

You MUST output a JSON object matching the CompressedState schema.

The recent_messages field MUST contain the most recent 2-4 message exchanges verbatim (as full message objects with role and content). These preserve conversational continuity.

The task_state fields MUST be concise but complete. Each string should be 1-2 sentences. Prefer specific identifiers (function names, file paths, line numbers) over vague descriptions.
"""


def _serialize_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"[{i}] {role}:\n{content}\n")
    return "\n".join(lines)


def build_distillation_prompt(
    messages: list[dict[str, str]],
    pending_query: str | None,
    max_context: int,
    current_tokens: int,
    verbatim_budget_used: int = 0,
) -> str:
    # Remaining budget after allocating for verbatim recent messages
    remaining_budget = max_context - verbatim_budget_used
    target_tokens = int(remaining_budget * settings.SEMANTIC_COMPRESSION_TARGET_PCT / 100)

    prompt = f"""Analyze the following conversation history and produce a compressed state.

CURRENT TOKEN COUNT: {current_tokens}
TARGET TOKEN COUNT: {target_tokens}
MAX ALLOWED: {max_context}

"""
    if pending_query:
        prompt += f"""PENDING USER QUERY: {pending_query}

IMPORTANT: Optimize the compression to preserve information specifically needed to answer the pending query. Background information irrelevant to this query may be discarded more aggressively.

"""
    prompt += f"""CONVERSATION HISTORY:
{_serialize_messages(messages)}

Produce a CompressedState that captures all information necessary to continue.
"""
    return prompt


def reconstruct_messages(state: object) -> list[dict[str, str]]:
    from .compression_schemas import CompressedState

    if not isinstance(state, CompressedState):
        raise TypeError(f"Expected CompressedState, got {type(state)}")

    parts = ["[Conversation State Summary]"]
    ts = state.task_state

    if ts.current_objective:
        parts.append(f"Objective: {ts.current_objective}")
    if ts.completed_steps:
        parts.append(f"Completed: {'; '.join(ts.completed_steps)}")
    if ts.pending_questions:
        parts.append(f"Pending: {'; '.join(ts.pending_questions)}")
    if ts.active_code_elements:
        parts.append(f"Active Code: {'; '.join(ts.active_code_elements)}")
    if ts.key_decisions:
        parts.append(f"Decisions: {'; '.join(ts.key_decisions)}")
    if ts.error_states:
        parts.append(f"Errors: {'; '.join(ts.error_states)}")
    if ts.tool_results_summary:
        parts.append(f"Tool Results: {ts.tool_results_summary}")
    if ts.user_preferences:
        parts.append(f"Preferences: {'; '.join(ts.user_preferences)}")

    summary_content = "\n".join(parts)
    messages: list[dict[str, str]] = [{"role": "system", "content": summary_content}]

    for msg in state.recent_messages:
        messages.append({"role": msg.role, "content": msg.content})

    return messages
