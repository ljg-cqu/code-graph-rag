# LLM-Based Semantic Context Compression — Architecture & Implementation Specification

## 1. Executive Summary

The current `ContextCompressor` is a rule-based text truncation engine masquerading as semantic compression. It reduces character counts but destroys information indiscriminately. This specification replaces the mechanical strategy pipeline with a **tiered compression architecture** whose core is an **LLM-driven information distillation step** that asks: *"What is the minimal information an LLM needs to continue this conversation successfully?"*

The new system preserves semantic intent, task state, decisions, errors, and pending actions while achieving superior token reduction compared to mechanical approaches.

---

## 2. Critical Analysis of Current System

### 2.1 The "Semantic" Misnomer

The existing 5 strategies (S1–S5) and the "semantic retention score" are entirely mechanical:

| Strategy | Actual Mechanism | Information Loss Pattern |
|----------|-----------------|------------------------|
| S1 Hierarchical Summarization | Hard character truncation to 200–300 chars | Arbitrary mid-sentence cuts; destroys context |
| S2 Stale Context Pruning | Keyword overlap with latest query | Drops messages that lack surface-level word matches but contain critical background |
| S3 Semantic Ranking Filter | Deduplication + word overlap ranking | Keeps messages with keyword matches even if redundant; drops semantically relevant messages without literal overlap |
| S4 Token-Aware Merging | Comment stripping + role merging | Destroys inline reasoning, code comments that explain *why* not just *what* |
| S5 Hybrid Summarization | Regex extraction of `def/class/error` keywords | Produces gibberish keyword lists with no semantic coherence |

**The retention score compounds the problem:** it measures regex-extracted entity overlap (`def hello`, `class World`) and keyword set intersection. A compressed result that says "User asked about function `hello`, assistant explained recursion" scores identically to one that contains the actual explanation. The score has **zero correlation** with whether the compressed context enables successful continuation.

### 2.2 Fundamental Architectural Flaws

1. **No model of "needed information"**: The compressor has no concept of *why* messages exist in context. It treats conversation history as a bag of text to be shortened.

2. **Strategies compete on mechanical metrics**: Token reduction % and regex entity overlap determine the "best" strategy. Neither metric measures whether the LLM can still answer follow-up questions.

3. **Rollback is adversarial**: When the mechanical retention score is low, the system rolls back to the original oversized context — which then fails at inference time due to token limits. The rollback itself is a bug, not a feature.

4. **Parallel strategy evaluation wastes CPU**: Five broken strategies run in parallel; the winner is the least broken. One correct strategy is superior to five incorrect ones.

### 2.3 Why Mechanical Compression Fails for Code RAG

In a codebase RAG conversation, context messages contain:
- **User intent evolution**: "Find the auth module" → "How does it handle tokens?" → "Why is refresh failing?"
- **Tool execution chains**: Query → results → follow-up query → more results
- **Code investigation paths**: File reads → function jumps → cross-reference lookups
- **Error recovery**: Failed assumptions, corrected understanding, new hypotheses

Mechanical truncation destroys the *chain of reasoning*. Keeping the last 4 messages preserves recency but drops the investigative path. Keyword overlap preserves surface matches but drops the semantic thread. Only an LLM can understand which links in the reasoning chain are load-bearing and which are redundant.

---

## 3. Design Principles

| Principle | Rationale |
|-----------|-----------|
| **P1 Information necessity over text reduction** | Compress to the *minimal sufficient context*, not the *maximal token reduction* |
| **P2 LLM judges information value** | Use an LLM to determine what matters because only an LLM understands semantic continuity |
| **P3 Structured distillation, not truncation** | Replace message sequences with structured state representations, not shorter versions of the same text |
| **P4 Tiered fallbacks guarantee progress** | Fast mechanical pre-processing → LLM distillation → hard truncation safety net |
| **P5 Query-aware compression** | When a pending query exists, optimize context for answering it specifically |
| **P6 Budget enforcement is invariant** | Output is *guaranteed* to fit within `max_context`; no rollback to oversized context |
| **P7 Observable and testable** | Compression decisions must be inspectable (strategy trace, rationale logging) |
| **P8 Sync/async compatibility** | Sync callers get mechanical compression; async callers get full LLM distillation |

---

## 4. Architecture

### 4.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: conversation history + optional pending query           │
│  Budget: max_context tokens                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 0: Budget Check                                           │
│  If total_tokens <= max_context: return no-op                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1: Mechanical Pre-Processing (lossless where possible)    │
│  - Deduplicate identical messages                               │
│  - Merge consecutive same-role messages with semantic separator │
│  - Drop empty/whitespace-only messages                          │
│  - Recompute token count                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 2: LLM Semantic Distillation (core innovation)            │
│  If still over budget: invoke LLM to produce structured         │
│  compression with explicit information categories               │
│  (Async only — sync callers skip to Tier 3)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 3: Hard Truncation Safety Net                             │
│  If LLM output still exceeds budget (rare): apply               │
│  deterministic message-level + content-level truncation         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output: compressed context + provenance metadata               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Diagram

```
┌──────────────────────┐      ┌──────────────────────────────┐
│  SemanticCompressor  │─────▶│  LLMContextDistiller         │
│  (orchestrator)      │      │  (PydanticAI Agent)          │
└──────────────────────┘      │  - system_prompt: distiller  │
         │                    │  - output_type: CompressedState │
         │                    └──────────────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────────┐      ┌──────────────────────────────┐
│  MechanicalPreProcessor      │  TokenBudgetEnforcer         │
│  (dedup, merge, clean)       │  (guarantees invariant P6)   │
└──────────────────────┘      └──────────────────────────────┘
```

---

## 5. Core Innovation: LLM Semantic Distillation

### 5.1 Information Model

Instead of shortening messages, the LLM extracts a `CompressedState` containing:

```python
class TaskState(BaseModel):
    current_objective: str          # What the user is trying to accomplish
    completed_steps: list[str]      # Key steps already taken/discovered
    pending_questions: list[str]    # Open questions or unresolved issues
    active_code_elements: list[str] # Functions/classes/files under discussion
    key_decisions: list[str]        # Architectural or approach decisions made
    error_states: list[str]         # Errors encountered and their resolution status
    tool_results_summary: str       # Summary of critical tool/ query outputs
    user_preferences: list[str]     # Constraints or preferences expressed

class Message(BaseModel):
    role: str
    content: str

class CompressedState(BaseModel):
    task_state: TaskState
    recent_messages: list[Message]  # Last N messages verbatim (chronological)
    compression_rationale: str      # Why the LLM kept what it kept
```

The `recent_messages` field preserves the most recent messages (typically last 2–4 exchanges) verbatim to maintain conversational continuity. The `task_state` captures everything older.

### 5.2 Why This Beats Truncation

**Example conversation:**
```
U: How does authentication work?
A: [reads auth.py, explains OAuth flow]
U: Why is refresh token failing?
A: [queries graph, finds bug in token validation]
U: Show me the exact line
A: [reads file, shows line 142]
U: What calls this function?
A: [runs cypher query, shows 3 callers]
U: Fix the bug
```

**Mechanical truncation (S5 hybrid):**
```
[5 older messages summarized]: Key points: function authenticate, class TokenManager,
function refresh, error validation, function validate_token
```

**LLM semantic distillation:**
```
Current objective: Fix a bug in refresh token validation (auth.py:142)
Completed steps: Identified OAuth flow, located validation bug in validate_token()
Pending: Apply the fix
Active code: auth.py:validate_token(), 3 callers found via cypher query
Key decisions: Bug is in validate_token() line 142, not in TokenManager
Error states: Refresh token fails due to incorrect validation logic (UNRESOLVED)
```

The mechanical result gives keywords without meaning. The LLM result gives **actionable state**.

### 5.3 Query-Aware Optimization

When a pending user query exists (the typical case — compression runs before generating the assistant response), the distillation prompt includes the query and instructs the LLM to prioritize information necessary to answer it.

**Without query:** Preserve general task state broadly.
**With query:** Preserve specifically the information needed to answer this query; it is acceptable to lose background that is irrelevant to the current question.

### 5.4 Verbatim Budget Management

A critical edge case: recent messages may themselves be very large (e.g., file contents, query results). Preserving 4 recent messages verbatim could exceed `max_context`.

**Solution:** Allocate a **verbatim budget** = `max_context * 0.4` (configurable). Before distillation:

1. Compute token count of messages designated as "recent"
2. If they exceed verbatim budget, truncate the *oldest* recent messages first (mechanically) until they fit
3. The remaining budget goes to `task_state`
4. The LLM distillation target is adjusted accordingly

This ensures the invariant that output fits within budget is never violated by verbatim preservation.

---

## 6. LLM Prompt Design

### 6.1 Distiller System Prompt

```
You are a context compression specialist. Your task is to analyze a conversation
history and produce a minimal representation that preserves all information
necessary for an LLM to continue the conversation successfully.

# Core Rule
You MUST NOT simply shorten or truncate messages. You must UNDERSTAND the
conversation, extract the semantic state, and reconstruct a minimal context
that an LLM can use to continue without loss of capability.

# Information Categories to Preserve

1. CURRENT OBJECTIVE: What is the user ultimately trying to accomplish?
2. COMPLETED STEPS: What investigative or implementation steps have been taken?
3. PENDING ISSUES: What questions are unanswered or problems unresolved?
4. ACTIVE CODE ELEMENTS: Which functions, classes, files are under discussion?
5. KEY DECISIONS: Any architectural choices, approach selections, or conclusions
6. ERROR STATES: Errors encountered, their causes, and whether they are resolved
7. CRITICAL TOOL RESULTS: Summaries of query results, file reads, or executions
   that changed the conversation state
8. USER PREFERENCES: Constraints, style preferences, or requirements expressed

# What You May Discard

- Redundant explanations of already-understood concepts
- Multiple rounds of clarification that converged to a single understanding
- Verbose tool output where a summary suffices
- Messages that are fully superseded by later messages
- Greetings, pleasantries, or meta-conversation

# Output Format

You MUST output a JSON object matching the CompressedState schema.

The `recent_messages` field MUST contain the most recent 2–4 message exchanges
verbatim (as full message objects with role and content). These preserve
conversational continuity.

The `task_state` fields MUST be concise but complete. Each string should be
1–2 sentences. Prefer specific identifiers (function names, file paths, line
numbers) over vague descriptions.
```

### 6.2 Distiller User Prompt

```python
def build_distillation_prompt(
    messages: list[dict[str, str]],
    pending_query: str | None,
    max_context: int,
    current_tokens: int,
) -> str:
    target_tokens = int(max_context * settings.SEMANTIC_COMPRESSION_TARGET_PCT / 100)

    prompt = f"""Analyze the following conversation history and produce a compressed state.

CURRENT TOKEN COUNT: {current_tokens}
TARGET TOKEN COUNT: {target_tokens}
MAX ALLOWED: {max_context}

"""
    if pending_query:
        prompt += f"""PENDING USER QUERY: {pending_query}

IMPORTANT: Optimize the compression to preserve information specifically needed
to answer the pending query. Background information irrelevant to this query
may be discarded more aggressively.

"""
    prompt += f"""CONVERSATION HISTORY:
{serialize_messages(messages)}

Produce a CompressedState that captures all information necessary to continue.
"""
    return prompt
```

### 6.3 Reconstruction to Message Format

The `CompressedState` is reconstructed into standard message format for the LLM:

```python
def reconstruct_messages(state: CompressedState) -> list[dict[str, str]]:
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
```

**Note on system message handling:** If the original context already contains system messages, they are preserved in the `preserved` set (Tier 1). The reconstruction's system message is appended after them. Multiple system messages are valid in most LLM APIs.

---

## 7. Implementation Specification

### 7.1 New Files

| File | Purpose |
|------|---------|
| `codebase_rag/semantic_compressor.py` | New LLM-based compressor (replaces strategy pipeline) |
| `codebase_rag/compression_prompts.py` | Distiller system prompt and user prompt builder |
| `codebase_rag/compression_schemas.py` | `TaskState`, `CompressedState`, `Message` Pydantic models |
| `codebase_rag/tests/test_semantic_compressor.py` | Unit tests for new compressor |
| `codebase_rag/tests/test_compression_integration.py` | Integration tests with mock LLM |

### 7.2 Modified Files

| File | Change |
|------|--------|
| `codebase_rag/context_compressor.py` | Add deprecation warnings to old strategies; add `compress_async()` wrapper that delegates to `SemanticCompressor` for backward compatibility |
| `codebase_rag/config.py` | Add `SEMANTIC_COMPRESSION_*` settings |
| `codebase_rag/services/llm.py` | Add `create_compression_agent()` factory |
| `codebase_rag/main.py` | Pass `pending_query` to compressor; update `/compress` command display; use `await compressor.compress()` |

### 7.3 Detailed Class Design

#### `SemanticCompressor`

```python
class SemanticCompressor:
    """Tiered context compressor with LLM semantic distillation."""

    def __init__(
        self,
        context: list[dict[str, Any]],
        max_context: int,
        pending_query: str | None = None,
        aggressive_mode: bool = False,
        preserve_pattern: str | None = None,
        agent: Agent | None = None,
    ):
        self.context = context
        self.max_context = max_context
        self.pending_query = pending_query
        self.aggressive_mode = aggressive_mode
        self.preserve_pattern = preserve_pattern
        self._agent = agent
        self.original_tokens = self._count_context_tokens(context)

    def compress_sync(self) -> CompressionResult:
        """Synchronous compression: mechanical tiers only.

        Does not invoke LLM distillation. Use compress() for full semantic compression.
        """
        return self._run_tiers(use_llm=False)

    async def compress(self) -> CompressionResult:
        """Asynchronous compression: full pipeline with LLM distillation."""
        return await self._run_tiers_async()

    def _run_tiers(self, use_llm: bool) -> CompressionResult:
        """Run tiers 0, 1, and optionally 3 (truncation)."""
        ...

    async def _run_tiers_async(self) -> CompressionResult:
        """Run tiers 0, 1, 2 (LLM), and 3 if needed."""
        ...
```

#### `LLMContextDistiller`

```python
class LLMContextDistiller:
    """Wraps PydanticAI agent for context distillation."""

    PROMPT_OVERHEAD_RESERVE: int = 2000  # Reserve for system prompt + formatting overhead

    def __init__(self, agent: Agent | None = None):
        self._agent = agent

    @property
    def _llm_input_cap(self) -> int:
        """Get the LLM input token cap from settings."""
        return settings.SEMANTIC_COMPRESSION_LLM_INPUT_CAP

    async def distill(
        self,
        messages: list[dict[str, Any]],
        pending_query: str | None,
        max_context: int,
    ) -> list[dict[str, Any]]:
        agent = self._agent or self._create_default_agent()
        current_tokens = count_context_tokens(messages)

        # Calculate effective cap accounting for prompt overhead
        effective_cap = self._llm_input_cap - self.PROMPT_OVERHEAD_RESERVE

        # Cap input to distillation LLM to prevent exceeding its own context window
        if current_tokens > effective_cap:
            messages = self._truncate_input_for_distillation(messages, effective_cap)
            current_tokens = count_context_tokens(messages)

        prompt = build_distillation_prompt(messages, pending_query, max_context, current_tokens)
        result = await agent.run(prompt)
        compressed_state: CompressedState = result.output
        return reconstruct_messages(compressed_state)

    def _truncate_input_for_distillation(
        self, messages: list[dict[str, Any]], cap: int
    ) -> list[dict[str, Any]]:
        """If input exceeds LLM cap, keep system + recent messages, summarize middle."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        recent = messages[-settings.SEMANTIC_COMPRESSION_MAX_RECENT_MESSAGES:]
        middle = messages[len(system_msgs):-len(recent)] if len(recent) < len(messages) else []

        # Truncate middle messages to fit within cap
        budget = cap - count_context_tokens(system_msgs + recent)
        truncated_middle = self._hard_truncate_to_budget(middle, max(0, budget))
        return system_msgs + truncated_middle + recent

    def _create_default_agent(self) -> Agent:
        config = getattr(settings, f"active_{settings.SEMANTIC_COMPRESSION_MODEL_ROLE}_config")
        llm = _create_provider_model(config)
        return Agent(
            model=llm,
            system_prompt=COMPRESSION_SYSTEM_PROMPT,
            output_type=CompressedState,
            retries=settings.AGENT_RETRIES,
        )
```

### 7.4 Mechanical Pre-Processing (Tier 1)

```python
def _mechanical_preprocess(
    self, context: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Lossless or low-loss mechanical reductions."""
    # Preserve system/tool/function messages
    preserved, compressible = self._preserve_matching_content(context)

    # Deduplicate identical messages in compressible set
    seen_hashes = set()
    deduped = []
    for msg in compressible:
        h = hash(json.dumps(msg, sort_keys=True))
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(msg)

    # Merge consecutive same-role messages
    merged = self._merge_consecutive_same_role(deduped)

    # Drop empty messages
    cleaned = [m for m in merged if str(m.get("content", "")).strip()]

    return preserved + cleaned
```

**Why keep this tier:** Deduplication and merging are lossless (or nearly so) and reduce token count without an LLM call. They handle common patterns like repeated tool result messages or fragmented assistant outputs.

### 7.5 Hard Truncation Safety Net (Tier 3)

The existing `_hard_truncate_to_budget()` and `_truncate_message_to_budget()` are retained as the final safety net, with one improvement: **the truncation preserves messages in their original order** (not reversed) to maintain chronological coherence, dropping from the middle/old portion first while keeping system messages and the most recent exchanges.

---

## 8. Configuration Additions

```python
# codebase_rag/config.py

# Context Window Compression Configuration
CONTEXT_COMPRESSION_SYSTEM_RESERVE_PCT: float = Field(
    default=15.0, gt=0, lt=50,
    description="Percentage of context window to reserve for system prompt, tools, and response buffer"
)

# Semantic Compression Configuration
SEMANTIC_COMPRESSION_ENABLED: bool = True
SEMANTIC_COMPRESSION_MODEL_ROLE: Literal["orchestrator", "cypher"] = "orchestrator"
SEMANTIC_COMPRESSION_MAX_RECENT_MESSAGES: int = Field(default=4, ge=1, le=10)
SEMANTIC_COMPRESSION_TARGET_PCT: float = Field(default=75.0, gt=0, lt=100)
SEMANTIC_COMPRESSION_TIMEOUT_SECONDS: int = Field(default=30, gt=0)
SEMANTIC_COMPRESSION_FALLBACK_ON_ERROR: bool = True
SEMANTIC_COMPRESSION_VERBATIM_BUDGET_PCT: float = Field(default=40.0, gt=0, lt=80)
SEMANTIC_COMPRESSION_LLM_INPUT_CAP: int = Field(default=50000, gt=1000)
SEMANTIC_COMPRESSION_PRESERVE_VERBATIM_ROLES: frozenset[str] = frozenset({"system", "tool", "function"})
```

| Setting | Default | Purpose |
|---------|---------|---------|
| `CONTEXT_COMPRESSION_SYSTEM_RESERVE_PCT` | `15.0` | Percentage of context window reserved for system prompt, tools, and response buffer |
| `SEMANTIC_COMPRESSION_ENABLED` | `True` | Master switch for LLM distillation |
| `SEMANTIC_COMPRESSION_MODEL_ROLE` | `"orchestrator"` | Which configured model to use for distillation |
| `SEMANTIC_COMPRESSION_MAX_RECENT_MESSAGES` | `4` | How many recent exchanges to preserve verbatim |
| `SEMANTIC_COMPRESSION_TARGET_PCT` | `75.0` | Target % of max_context for distillation output |
| `SEMANTIC_COMPRESSION_TIMEOUT_SECONDS` | `30` | LLM call timeout |
| `SEMANTIC_COMPRESSION_FALLBACK_ON_ERROR` | `True` | Whether to fall back to truncation if LLM fails |
| `SEMANTIC_COMPRESSION_VERBATIM_BUDGET_PCT` | `40.0` | Max % of budget allocated to verbatim recent messages |
| `SEMANTIC_COMPRESSION_LLM_INPUT_CAP` | `50000` | Max tokens sent to distillation LLM (protects LLM context window) |

---

## 9. Integration Points

### 9.1 Main Loop Integration (`main.py`)

The compression trigger currently runs before the RAG agent:

```python
# Current (main.py ~674-770)
if settings.CONTEXT_COMPRESSION_ENABLED and message_history:
    serialized = "\n".join([str(m) for m in message_history] + [question_with_context])
    total_tokens = count_tokens(serialized)
    max_context = settings.DEFAULT_CONTEXT_WINDOW
    trigger_threshold = int(max_context * settings.CONTEXT_COMPRESSION_AUTO_TRIGGER_PCT / 100)
    if total_tokens >= trigger_threshold:
        compressor = ContextCompressor(context=context, max_context=max_context, ...)
        result = compressor.compress_sync()
        message_history[:] = compressed_history
```

**Updated integration:**

```python
if settings.CONTEXT_COMPRESSION_ENABLED and message_history:
    total_tokens = count_context_tokens(message_history)
    
    # Get model context window
    model_context_window = get_model_context_window()
    
    # Apply system reserve for system prompt, tools, and response buffer
    system_reserve_factor = (100 - settings.CONTEXT_COMPRESSION_SYSTEM_RESERVE_PCT) / 100
    max_context = int(model_context_window * system_reserve_factor)
    
    trigger_threshold = int(max_context * settings.CONTEXT_COMPRESSION_AUTO_TRIGGER_PCT / 100)
    if total_tokens >= trigger_threshold:
        compressor = SemanticCompressor(
            context=message_history,
            max_context=max_context,
            pending_query=question_with_context,  # NEW: pass pending query
        )
        result = await compressor.compress()
        message_history[:] = result.compressed_context
```

### 9.2 Manual `/compress` Command

Update the `/compress` command to:
1. Display whether compression used mechanical, semantic, or truncation tier
2. Show the `compression_rationale` if semantic compression was used
3. Display `task_state` fields in a structured table

---

## 10. Testing Strategy

### 10.1 Unit Tests

| Test | What It Validates |
|------|-------------------|
| `test_no_op_when_under_budget` | Tier 0 fast path |
| `test_mechanical_preprocess_dedupes` | Tier 1 deduplication |
| `test_mechanical_preprocess_merges_same_role` | Tier 1 merging |
| `test_compress_sync_skips_llm` | Sync API does not invoke LLM |
| `test_compress_async_uses_llm` | Async API invokes LLM when needed |
| `test_llm_distill_produces_valid_schema` | Tier 2 output schema validation |
| `test_llm_distill_preserves_recent_messages` | Tier 2 verbatim preservation |
| `test_llm_distill_includes_task_state` | Tier 2 state extraction |
| `test_hard_truncation_enforces_budget` | Tier 3 invariant |
| `test_fallback_on_llm_error` | Error handling |
| `test_query_aware_prioritizes_relevant_info` | Query-aware optimization |
| `test_verbatim_budget_respected` | Recent messages don't exceed budget |
| `test_llm_input_cap_enforced` | Large inputs are pre-truncated |

### 10.2 Integration Tests

| Test | What It Validates |
|------|-------------------|
| `test_end_to_end_compression_reduces_tokens` | Full pipeline reduces tokens |
| `test_compressed_context_answers_followup` | LLM can answer follow-up from compressed context |
| `test_compression_with_real_llm` | Live distillation (marked `@pytest.mark.slow`) |

### 10.3 Mock Strategy

Use `unittest.mock.AsyncMock` to mock the PydanticAI agent's `run()` method, returning pre-constructed `CompressedState` objects. This allows testing the pipeline logic without LLM calls.

---

## 11. Implementation Phases

### Phase 1: Foundation (1–2 days)
1. Create `compression_schemas.py` with `TaskState`, `CompressedState`, `Message`
2. Create `compression_prompts.py` with system prompt and builder
3. Create `SemanticCompressor` class with Tier 0, Tier 1, Tier 3
4. Add configuration settings
5. Write unit tests for tiers 0, 1, 3
6. Add `compress_async()` to `ContextCompressor` that delegates to `SemanticCompressor`

### Phase 2: LLM Integration (1–2 days)
1. Implement `LLMContextDistiller` with PydanticAI agent
2. Add `create_compression_agent()` to `services/llm.py`
3. Wire Tier 2 into `SemanticCompressor`
4. Write unit tests with mocked agent
5. Write integration test with real LLM (marked slow)

### Phase 3: Integration & Migration (1 day)
1. Update `main.py` compression trigger to use `await SemanticCompressor.compress()`
2. Update `/compress` command display
3. Add deprecation warnings to old strategy methods in `ContextCompressor`
4. Update existing tests in `test_context_compressor.py` to verify backward compatibility

### Phase 4: Validation (1 day)
1. Run full test suite: `make test`
2. Run integration tests: `make test-integration`
3. Manual validation: start chat, build long context, verify compression
4. Measure token reduction vs. old system on sample conversations

---

## 12. Risk Analysis & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM distillation is too slow (adds latency) | Medium | High | Timeout + fallback; mechanical tier handles most cases; LLM only when needed |
| LLM produces oversized output | Low | High | Reconstruct and measure tokens; verbatim budget prevents recent messages from exceeding; if still over, truncate summary text |
| LLM drops critical information | Medium | High | Preserve recent messages verbatim; structured schema guides the LLM; human-readable rationale for debugging |
| Cost of extra LLM call | Medium | Medium | Only triggered when context exceeds threshold (~85% of budget); mechanical tier reduces frequency; use cheaper model role if configured |
| Backward compatibility breaks | Low | High | Keep `CompressionResult` API identical; `compress_sync()` remains sync; `compress()` is new async method |
| Distillation input exceeds LLM context window | Low | High | `SEMANTIC_COMPRESSION_LLM_INPUT_CAP` limits input size; `PROMPT_OVERHEAD_RESERVE` accounts for formatting; pre-truncation strategy for very large histories |

---

## 13. Comparison: Old vs. New

| Aspect | Old System | New System |
|--------|-----------|------------|
| Core mechanism | 5 mechanical strategies | LLM semantic distillation + structured state |
| Information model | None (text reduction) | `TaskState` with objective, decisions, errors |
| Retention measurement | Regex entity + keyword overlap | Not needed — LLM preserves meaning by construction |
| Rollback behavior | Reverts to oversized context (breaks inference) | Never rolls back to oversized context; hard truncation guarantees budget |
| Query awareness | None (S2/S3 use keyword overlap as proxy) | Explicit — pending query guides distillation priority |
| Parallelism | 5 strategies in thread pool | Single LLM call (or mechanical tier, no parallelism needed) |
| Observability | Opaque score number | Human-readable `compression_rationale` |
| Verbatim preservation | None (all messages subject to truncation) | Recent messages preserved verbatim; system/tool preserved |
| Sync/async | `compress_sync()` only | `compress_sync()` = mechanical; `compress()` = full semantic |

---

## 14. Files Summary

### New Files
- `.specs/llm_semantic_compression_spec.md` (this document)
- `codebase_rag/compression_schemas.py`
- `codebase_rag/compression_prompts.py`
- `codebase_rag/semantic_compressor.py`
- `codebase_rag/tests/test_semantic_compressor.py`
- `codebase_rag/tests/test_compression_integration.py`

### Modified Files
- `codebase_rag/context_compressor.py` — add deprecation warnings; add backward-compatible async wrapper
- `codebase_rag/config.py` — add semantic compression settings
- `codebase_rag/services/llm.py` — add compression agent factory
- `codebase_rag/main.py` — pass pending_query, update /compress display, use async compress
- `codebase_rag/tests/test_context_compressor.py` — update for new behavior

---

*Specification version: 1.0*
*Last updated: 2026-04-18*
*Target: codebase_rag v0.x*
*Status: Draft — awaiting review*
