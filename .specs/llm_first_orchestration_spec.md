# LLM-First Orchestration Design Spec

## Executive Summary

The codebase currently relies on extensive rule-based Python logic (regex patterns, keyword matching, hardcoded if/elif chains, threshold-based overrides) to make semantic decisions that should be delegated to the LLM. This creates a **cascade of false positives** where brittle regex layers reject valid queries before the LLM ever sees them, and hardcoded mappings select suboptimal query methods based on superficial keyword matches.

**Core principle:** The LLM is the only component capable of understanding user intent with the necessary nuance. Python code should **provide tools and context**, not **make semantic judgments**. When a decision requires understanding natural language, context, or user goals, it must be made by the LLM via structured prompting.

**Rule-based code is acceptable only for:**
- Mechanical transformations (parsing, serialization, protocol adapters)
- Safety guardrails with zero false-positive tolerance (e.g., blocking `DROP DATABASE` in Cypher)
- Performance optimizations that do not alter semantic outcomes (caching, batch sizing)

This spec replaces 14 major rule-based subsystems with LLM-driven equivalents.

---

## Prerequisites

Before implementing this spec, ensure the following configurations exist in `codebase_rag/config.py`:

```python
# Orchestrator LLM configuration (add if not present)
ACTIVE_ORCHESTRATOR_PROVIDER: str = "openai"  # or "anthropic", "google", "local"
ACTIVE_ORCHESTRATOR_MODEL: str = "gpt-4o-mini"  # Fast model for planning
```

If using a unified model config, ensure `settings.active_orchestrator_config` exists and is used for orchestration/planning. `settings.active_cypher_config` should only be used for Cypher generation/repair.

---

## Problem Analysis

### Anti-Pattern Catalog

#### 1. Query Intent Classification via Keyword Matching
**Files:** `codebase_rag/retrieval/query_orchestrator.py:105-145`

**Current code:**
```python
_INTENT_KEYWORD_MAP: dict[QueryIntent, frozenset[str]] = {
    QueryIntent.STRUCTURAL: frozenset({"call", "calls", "caller", "hierarchy", "inherit", "import", "depend"}),
    QueryIntent.FUNCTIONAL: frozenset({"how does", "how to", "what does", "explain", "work", "behavior"}),
    QueryIntent.VALIDATION: frozenset({"correct", "valid", "implement", "comply", "spec", "should", "must", "require"}),
    QueryIntent.SEMANTIC: frozenset({"similar", "like", "compare", "related", "same as", "function"}),
}
```

**Failure cases:**
- `"What functions implement the validation spec?"` matches **4 intents simultaneously** (STRUCTURAL via "functions", FUNCTIONAL via "What", VALIDATION via "validation"/"spec", SEMANTIC via "function"). The code arbitrarily picks the first match.
- `"Show me the call hierarchy for the authentication module"` is STRUCTURAL, but `"How does the call stack work?"` is FUNCTIONAL — both contain "call".

#### 2. Query Method Selection via Hardcoded Intent→Method Map
**Files:** `codebase_rag/retrieval/query_orchestrator.py:148-169,438-447`

**Current code:**
```python
_INTENT_METHOD_MAP: dict[QueryIntent, tuple[list[QueryMethod], list[QueryMethod]]] = {
    QueryIntent.FUNCTIONAL: (
        [QueryMethod.SEMANTIC_SEARCH, QueryMethod.GRAPH_TRAVERSAL],
        [QueryMethod.KEYWORD_SEARCH, QueryMethod.VECTOR_DIRECT],
    ),
    QueryIntent.STRUCTURAL: (
        [QueryMethod.GRAPH_TRAVERSAL, QueryMethod.GRAPH_NAVIGATION],
        [QueryMethod.KEYWORD_SEARCH, QueryMethod.SEMANTIC_SEARCH],
    ),
    ...
}
```

**Failure cases:**
- The mapping assumes all FUNCTIONAL queries should start with SEMANTIC_SEARCH, but `"What is the shortest path from A to B?"` is FUNCTIONAL yet needs GRAPH_ALGORITHMS.
- The confidence threshold (`0.5`) is arbitrary. A misclassified intent with confidence `0.51` gets its "primary" methods, while `0.49` triggers an expensive run of ALL methods.

#### 3. Template-Based Cypher Fallback with Keyword Matching
**Files:** `codebase_rag/retrieval/query_orchestrator.py:603-669`, `codebase_rag/cypher_queries.py:171-236`

**Current code:**
```python
if any(kw in query_lower for kw in ["call", "caller", "calls", "called by"]):
    templates_to_try.append(("find_callers_of", {"qn": keyword}))
if any(kw in query_lower for kw in ["import", "importer", "imports"]):
    templates_to_try.append(("find_importers_of", {"qn": keyword}))
if any(kw in query_lower for kw in ["depend", "uses", "reference"]):
    templates_to_try.append(("find_dependencies", {"keyword": keyword, "limit": top_k}))
```

**Failure cases:**
- `"Which modules import the authentication service?"` might match both "import" and "depend" patterns; template selection order is arbitrary.
- `"Can you call out the functions that are never called?"` matches "call" → `find_callers_of`, but the user wants an analysis of uncalled functions, not caller lookup.
- `keyword` is extracted via `extract_best_keyword()` (longest non-stopword), which is often wrong (e.g., `"modules"` may be longest instead of `"authentication"`).

#### 4. Graph Algorithm Selection via Keyword Matching
**Files:** `codebase_rag/retrieval/query_orchestrator.py:918-978`

**Current code:**
```python
if any(word in query_lower for word in ["similar", "like", "same as", "analogous"]):
    similar_nodes = await asyncio.to_thread(algo.get_similar_nodes, node_id, top_k)
elif any(word in query_lower for word in ["context", "neighbors", "around", "surrounding"]):
    bfs_nodes = await asyncio.to_thread(algo.get_bfs_context, node_id, 3)
else:
    similar_nodes = await asyncio.to_thread(algo.get_similar_nodes, node_id, top_k)
```

**Failure cases:**
- `"Functions with similar names but different behaviors"` contains "similar" → triggers structural Jaccard similarity, but the user wants semantic comparison.
- `"What is the surrounding context of this error?"` contains "surrounding" → triggers BFS, but the user may want semantic context, not graph neighbors.

#### 5. Explicit Parallel/Sequential Pattern Matching in Eligibility Classifier
**Files:** `codebase_rag/orchestrator/concurrency_eligibility_classifier.py:44-70,388-418`

**Current code:**
```python
EXPLICIT_PARALLEL_PATTERNS = [
    r"run in parallel", r"use parallel execution", r"split into parallel subtasks",
    r"parallelize this", r"execute in parallel",
]
EXPLICIT_SEQUENTIAL_PATTERNS = [
    r"run sequentially", r"no parallel", r"run one at a time",
    r"sequential execution", r"do not parallelize",
]
SAFETY_NON_ELIGIBLE_PATTERNS = [
    (r"single file", 0.95), (r"one (file|function|class)", 0.9),
    (r"(write|modify|delete|update|create) (file|code|config|document)", 0.9),
]
```

**Failure cases:**
- `"Please parallelize this analysis but run the final merge sequentially"` matches both parallel and sequential patterns. The code checks sequential first, so it blocks — but the user's intent is nuanced.
- `"Can you create a summary of the single file?"` matches `r"single file"` → blocked as non-eligible, but creating a summary is read-only.

#### 6. Conceptual Question Detection via Regex
**Files:** `codebase_rag/orchestrator/concurrency_eligibility_classifier.py:324-342`

**Current code:**
```python
def _is_conceptual_question(self, prompt: str) -> bool:
    conceptual_patterns = [
        r"\b(what is|what are|why|how to|explain|describe|what does)\b",
        r"\b(importance of|benefits of|purpose of|meaning of)\b",
        r"\b(categorical thinking|concept|theory|framework|methodology)\b",
    ]
    file_patterns = [
        r"\b(file|function|class|method)\s+\w+",
        r"\b(in|from)\s+[\w/]+\.(py|js|ts|java|cpp)\b",
        r"```[\w/]+```",
    ]
    has_conceptual = any(re.search(p, prompt, re.I) for p in conceptual_patterns)
    has_file_ref = any(re.search(p, prompt, re.I) for p in file_patterns)
    return has_conceptual and not has_file_ref
```

**Failure cases:**
- `"Explain what the auth module does in main.py"` is a conceptual question WITH a file reference → returns False, so it may be parallelized across files incorrectly.
- `"What is the function signature of create_user?"` has no file reference but is NOT conceptual → returns True, so it's treated as conceptual and blocked from parallelization.

#### 7. Write Intent Detection via Multi-Layer Regex
**Files:** `codebase_rag/main.py:1717-1823`

**Current code:**
A 4-layer regex system with explicit_write_patterns, read_only_context_patterns, ambiguous_write_patterns, and a default safe layer.

**Failure cases:**
- `"Fix the bug in the authentication module"` matches explicit write pattern → blocked, but the user may only want an analysis of the bug.
- `"Review the update logic in the scheduler"` matches read-only context pattern, but what if they actually want to modify it afterward?
- Layer 4 defaults to False (read-only), but Layer 1 is overly broad and catches many read-only queries.

#### 8. Task Splitting Strategy Detection via Keyword Matching
**Files:** `codebase_rag/orchestrator/task_splitter.py:113-153`

**Current code:**
```python
def _detect_strategy(self, prompt: str) -> str:
    lower_prompt = prompt.lower()
    file_indicators = ["all files", "each file", "files in", "directory", "folder", "review code", "scan files", "generate tests for", "analyze files"]
    if any(indicator in lower_prompt for indicator in file_indicators):
        return "file"
    node_indicators = ["all functions", "all classes", "all methods", "function definitions", "class declarations", "interface definitions"]
    if any(indicator in lower_prompt for indicator in node_indicators):
        return "node"
    return "file"
```

**Failure cases:**
- `"Generate tests for the authentication functions in all files"` matches both file and node indicators. Only the first check wins.
- `"Review code architecture across all modules"` contains "review code" → file-based splitting, but the user wants architectural analysis, not per-file review.

#### 9. Task Complexity Scoring via Keyword Matching
**Files:** `codebase_rag/orchestrator/task_splitter.py:173-200`

**Current code:**
```python
simple_keywords = ["count", "list", "filter", "sort", "find", "search", "check", "verify"]
complex_keywords = ["analyze", "generate", "refactor", "explain", "design", "implement", "debug", "fix", "review"]

complexity = 2
if any(k in lower_prompt for k in simple_keywords):
    complexity = 1
if any(k in lower_prompt for k in complex_keywords):
    complexity = 4
```

**Failure cases:**
- `"Find and fix all race conditions in the codebase"` matches both "find" (simple=1) and "fix" (complex=4). The scoring doesn't accumulate, it just overwrites.
- `"Review this 10-line function"` matches "review" → complexity 4, but it's trivial.

#### 10. File Type Hint Extraction via Keyword Matching
**Files:** `codebase_rag/orchestrator/task_splitter.py:520-591`

**Current code:**
```python
if any(word in lowered for word in ["python", ".py", "django", "flask"]):
    extension_hints.append(".py")
if any(word in lowered for word in ["javascript", ".js", "react", "node", "express"]):
    extension_hints.extend([".js", ".jsx", ".ts", ".tsx"])
if "test" in lowered or "spec" in lowered:
    name_pattern_hints.extend(["_test", "_spec", "test_", "spec_", ".test", ".spec"])
```

**Failure cases:**
- `"How does Node.js handle Python subprocesses?"` matches both `.py` and `.js`.
- `"Show me the configuration for the express middleware"` matches "express" → includes ALL JS/TS files, even in a Go project where "express" might be a package name.

#### 11. Scope Path Extraction from Prompt via Regex
**Files:** `codebase_rag/orchestrator/task_splitter.py:327-353`

**Current code:**
```python
candidates.extend(re.findall(r"['\"]([^'\"]+)['\"]", prompt))
path_patterns = [
    r"(?:in|under|within|inside|from|at)\s+([A-Za-z0-9_./\\-]+)",
    r"(?:file|files|folder|directory|path|paths)\s+(?:in|under|within|inside|from|at)?\s*([A-Za-z0-9_./\\,-]+)",
]
candidates.extend(re.findall(r"(?:\.{0,2}/)?[A-Za-z0-9_./\\-]+", prompt))
```

**Failure cases:**
- Any quoted string becomes a candidate path. `"Explain how 'async def' works in Python"` extracts `async def` as a path.
- `"What is the path from user input to database in this app?"` extracts `user input to database` as a scope path.

#### 12. Question Type Classification for Sufficiency Analysis
**Files:** `codebase_rag/orchestrator/sufficiency_analyzer.py:23-68`

**Current code:**
```python
_DIAGNOSTIC_KEYWORDS = frozenset({"why is", "why does", "what is wrong", "debug", "failing", "error", "issue", "problem", "not working", "broken", "stack trace", "exception", "crash"})
_FUNCTIONAL_KEYWORDS = frozenset({"how does", "how do", "implementation", "logic", "algorithm", "work", "flow", "behavior", "what happens when", "step by step", "describe", "explain", "process", "mechanism"})
_STRUCTURAL_KEYWORDS = frozenset({"what classes", "what functions", "list all", "show me", "find all", "how many", "count", "directory", "structure", "hierarchy", "dependencies", "relationships"})

def analyze_requirements(question: str) -> InvestigationRequirements:
    q_lower = question.lower()
    if any(kw in q_lower for kw in _DIAGNOSTIC_KEYWORDS):
        return InvestigationRequirements(QuestionType.DIAGNOSTIC, requires_vector=True, requires_graph=True, requires_file_read=True, min_rounds=3, requires_cross_validation=True)
    if any(kw in q_lower for kw in _FUNCTIONAL_KEYWORDS):
        return InvestigationRequirements(QuestionType.FUNCTIONAL, requires_vector=True, requires_graph=True, requires_file_read=True, min_rounds=3)
    return InvestigationRequirements(QuestionType.STRUCTURAL, requires_vector=True, requires_graph=True, requires_file_read=False, min_rounds=2)
```

**Failure cases:**
- `"Why does the error handler use a retry mechanism?"` matches DIAGNOSTIC ("why does", "error") but is actually FUNCTIONAL.
- `"List all exceptions and explain why they crash"` matches both STRUCTURAL ("list all") and DIAGNOSTIC ("crash", "why").
- `min_rounds=3` for DIAGNOSTIC/FUNCTIONAL is arbitrary. Some diagnostic questions are trivial; some structural questions need deep investigation.

#### 13. Tool Filtering Based on Hardcoded Mode→Tool Map
**Files:** `codebase_rag/tools/__init__.py:20-56`

**Current code:**
```python
_CODE_ONLY_TOOL_NAMES = {"validate_code_against_spec"}
_DOCUMENT_ONLY_TOOL_NAMES = {"query_document_graph"}
_BOTH_MERGED_TOOL_NAMES = {"query_both_graphs", "validate_doc_against_code"}

def get_tools_for_mode(mode: QueryMode, tools: list[Tool]) -> list[Tool]:
    if mode == QueryMode.BOTH_MERGED:
        return tools
    filtered = []
    for tool in tools:
        name = getattr(tool, "name", "")
        if name in _BOTH_MERGED_TOOL_NAMES:
            continue
        if mode == QueryMode.DOCUMENT_ONLY and name in _CODE_ONLY_TOOL_NAMES:
            continue
        if mode == QueryMode.CODE_ONLY and name in _DOCUMENT_ONLY_TOOL_NAMES:
            continue
        filtered.append(tool)
    return filtered
```

**Failure cases:**
- In DOCUMENT_ONLY mode, `validate_code_against_spec` is removed. But what if the user asks: `"Does the documentation spec match the actual code?"` — they need a validation tool.
- In CODE_ONLY mode, `query_document_graph` is removed. But the user might ask: `"What does the README say about this function?"` — they need document access.
- The LLM agent is forced to work with a mutilated toolset based on an arbitrary mode.

#### 14. Cypher Read-Only Validation via Dangerous Keyword Blacklist
**Files:** `codebase_rag/services/llm.py:179-234`

**Current code:**
```python
_CYPHER_DANGEROUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (kw, _build_keyword_pattern(kw)) for kw in cs.CYPHER_DANGEROUS_KEYWORDS
]

def _validate_cypher_read_only(query: str) -> None:
    upper_query = query.upper()
    if upper_query.count(";") > 1:
        raise ex.LLMGenerationError(...)
    if "OVER(" in upper_query:
        raise ex.LLMGenerationError(...)
    for keyword, pattern in _CYPHER_DANGEROUS_PATTERNS:
        if pattern.search(upper_query):
            raise ex.LLMGenerationError(...)
    for match in _CALL_PROCEDURE_PATTERN.finditer(query):
        procedure = match.group("proc").lower()
        is_safe = any(procedure == safe or procedure.startswith(safe + ".") for safe in cs.CYPHER_SAFE_CALL_PROCEDURES)
        if not is_safe:
            raise ex.LLMGenerationError(...)
```

**Failure cases:**
- `"MATCH (n:User) WHERE n.role = 'admin' RETURN n.name"` might trigger a dangerous keyword if `"DELETE"` appears in a string literal or comment.
- The whitelist approach for `CALL` procedures is maintenance-heavy. Every new Memgraph plugin requires a code change.
- `"MATCH (n) SET n.analyzed = true RETURN n"` might not match any dangerous keyword but is clearly a write operation.

#### 15. Keyword Extraction for Cypher Queries
**Files:** `codebase_rag/utils/query_utils.py:11-67`

**Current code:**
```python
STOPWORDS = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'what', 'how', 'where', 'when', 'why', 'who', 'show', 'find', 'tell', 'get', 'list', 'all', 'any', 'some', 'does', 'do', 'can', 'could', 'would', 'should', 'will'}

def extract_best_keyword(query: str) -> str:
    words = query.lower().split()
    meaningful = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    if not meaningful:
        return max(words, key=len, default="")
    return max(meaningful, key=len)
```

**Failure cases:**
- For `"What is the authentication middleware for the API gateway?"`, the longest meaningful word is `authentication` — but the actual entity might be `api_gateway_auth`.
- For `"How does the shortest path algorithm work?"`, it extracts `shortest` — but the graph might store it as `dijkstra` or `astar`.
- Splitting on whitespace fails for multi-word entities (`"red black tree"` → `"black"`).

#### 16. Query Mode Auto-Detection Based on Entity Counts
**Files:** `codebase_rag/main.py:2952-2977`

**Current code:**
```python
def _determine_default_query_mode(code_graph, doc_graph) -> QueryMode:
    code_count, doc_count = _get_content_availability(code_graph, doc_graph)
    if code_count == 0 and doc_count > 0:
        return QueryMode.DOCUMENT_ONLY
    elif code_count > 0 and doc_count == 0:
        return QueryMode.CODE_ONLY
    elif code_count > 0 and doc_count > 0:
        return QueryMode.BOTH_MERGED
    else:
        return QueryMode.CODE_ONLY
```

**Failure cases:**
- If a repo has both code and docs but the user's query is purely about code (e.g., `"What functions call authenticate_user?"`), BOTH_MERGED is selected, which queries the document graph unnecessarily.
- If docs exist but are minimal (e.g., just a README), BOTH_MERGED is still selected.

---

## Proposed Solution: LLM-First Architecture

### Guiding Principles

1. **LLM decides semantics.** Python code never classifies intent, selects methods, or filters tools based on keyword matching.
2. **Tools are transparent.** Every tool has a clear description. The LLM sees all available tools and decides which to use.
3. **Structured outputs.** The LLM returns typed decisions (Pydantic models) that Python code executes mechanically.
4. **Fast-path for mechanical work.** Caching, batching, and protocol adapters remain rule-based. Semantic decisions do not.
5. **Mode is a hint, not a filter.** Query mode suggests a default graph but does not hide tools from the LLM.

---

### Part 1: LLM-Driven Query Orchestration

#### 1.1 Replace Intent Classification

**Remove:** `_INTENT_KEYWORD_MAP`, `classify_intent()`, `classify_intent_with_confidence()` from `query_orchestrator.py`.

**Replace with:** `LLMQueryPlanner` that outputs a structured plan.

**New file:** `codebase_rag/orchestrator/llm_query_planner.py`

```python
from __future__ import annotations

import hashlib
import time
from enum import StrEnum
from functools import lru_cache

from loguru import logger
from pydantic import BaseModel, Field

from codebase_rag.config import settings


class QueryMethod(StrEnum):
    SEMANTIC_SEARCH = "semantic_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    KEYWORD_SEARCH = "keyword_search"
    VECTOR_DIRECT = "vector_direct"
    GRAPH_ALGORITHMS = "graph_algorithms"
    GRAPH_NAVIGATION = "graph_navigation"


class GraphAlgorithm(StrEnum):
    """Specific graph algorithm to execute when method is GRAPH_ALGORITHMS."""

    SHORTEST_PATH = "shortest_path"
    SIMILARITY = "similarity"
    COMMUNITY = "community"
    BFS = "bfs"
    NONE = "none"


class QueryIntent(StrEnum):
    """Unified intent enum for both code and document queries.

    NOTE: This replaces the existing QueryIntent in
    codebase_rag/retrieval/query_orchestrator.py (which used Enum with auto()).
    Move this definition to codebase_rag/shared/query_router.py or
    codebase_rag/constants.py so both orchestrator and retrieval modules
    can import a single source of truth.
    """
    # Code graph intents
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    SEMANTIC = "semantic"
    EXPLORATORY = "exploratory"
    VALIDATION = "validation"
    # Document graph intents
    DOC_SEMANTIC_SEARCH = "doc_semantic_search"
    DOC_GRAPH_TRAVERSAL = "doc_graph_traversal"
    DOC_COMPARISON = "doc_comparison"
    DOC_PROCEDURAL = "doc_procedural"


class QueryPlan(BaseModel):
    """Structured query plan produced by the LLM.

    Uses Pydantic BaseModel for compatibility with pydantic_ai structured output.
    """

    methods: list[QueryMethod] = Field(
        ...,
        description="Ordered list of methods to execute (primary first, max 3).",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation for method selection.",
    )
    fallback_methods: list[QueryMethod] = Field(
        default_factory=list,
        description="Methods to try if primary methods fail or return empty.",
    )
    expected_entities: list[str] = Field(
        default_factory=list,
        description="Key entities (functions, classes, concepts) identified from the query.",
    )
    requires_file_read: bool = Field(
        default=False,
        description="Whether the query likely requires reading source files.",
    )
    intent: QueryIntent | None = Field(
        default=None,
        description="Classified intent type for routing decisions.",
    )
    algorithm: GraphAlgorithm = Field(
        default=GraphAlgorithm.NONE,
        description="Specific graph algorithm when method is graph_algorithms.",
    )


# TTL cache for validated plans (query hash -> (plan, timestamp))
# Caches successful LLM planning results to reduce API calls.
# Eviction is time-based (TTL) plus size-based (oldest timestamp removed).
_PLAN_CACHE: dict[str, tuple[QueryPlan, float]] = {}
_PLAN_CACHE_TTL = 3600  # 1 hour TTL
_PLAN_CACHE_MAX_SIZE = 1000


class LLMQueryPlanner:
    """Uses LLM to plan query execution instead of keyword-based intent classification.

    Implements lazy initialization and caching for efficiency.
    """

    __slots__ = ("agent", "_model_id")

    SYSTEM_PROMPT = """You are a query planner for a codebase analysis system. Given a user's natural language query and the available query methods, select the best methods to execute.

Available methods:
- semantic_search: Find code entities by meaning (embeddings). Best for "what does X do?", "find code about Y".
- graph_traversal: Follow relationships in the graph (calls, imports, inherits). Best for "who calls X?", "what imports Y?".
- keyword_search: Fast text matching on names and qualified_names. Best for exact name lookups.
- vector_direct: Direct vector similarity without graph. Best for finding similar code snippets.
- graph_algorithms: Run algorithms (shortest path, similarity, community). Best for "path from A to B", "most central function".
- graph_navigation: Browse graph neighborhoods. Best for "show me around X", "context of Y".

Available graph algorithms (only when method is graph_algorithms):
- shortest_path: Find shortest path between two entities.
- similarity: Find structurally or semantically similar nodes.
- community: Detect communities or clusters.
- bfs: Breadth-first search neighborhood context.
- none: No specific algorithm.

Respond with a JSON object matching this structure:
{
  "methods": ["method_name", ...],
  "reasoning": "brief explanation",
  "fallback_methods": ["method_name", ...],
  "expected_entities": ["entity_name", ...],
  "requires_file_read": true/false,
  "intent": "intent_type",
  "algorithm": "algorithm_name"
}

Rules:
- Select 1-3 primary methods. Do not select more than 3.
- Use fallback_methods for methods to try if primary methods return empty.
- expected_entities should list specific functions, classes, or modules mentioned.
- intent should be one of: structural, functional, semantic, exploratory, validation, doc_semantic_search, doc_graph_traversal, doc_comparison, doc_procedural.
- algorithm should be one of: shortest_path, similarity, community, bfs, none. Only set when "graph_algorithms" is in methods.
"""

    def __init__(self) -> None:
        self.agent = None
        self._model_id: str | None = None

    def _initialize_agent(self) -> None:
        """Lazy initialization of the LLM agent."""
        if self.agent is not None:
            return

        # Use orchestrator config (fast/cheap model for planning).
        # Falls back to cypher config only if orchestrator config is unavailable.
        try:
            config = settings.active_orchestrator_config
        except AttributeError:
            try:
                config = settings.active_cypher_config
            except AttributeError:
                # Fallback: create a minimal config
                from codebase_rag.config import ModelConfig
                config = ModelConfig(
                    provider=getattr(settings, "ACTIVE_ORCHESTRATOR_PROVIDER", "openai"),
                    model_id=getattr(settings, "ACTIVE_ORCHESTRATOR_MODEL", "gpt-4o-mini"),
                )

        from codebase_rag.providers import _create_provider_model

        llm = _create_provider_model(config)
        self._model_id = config.model_id

        from pydantic_ai import Agent

        self.agent = Agent(
            model=llm,
            system_prompt=self.SYSTEM_PROMPT,
            output_type=QueryPlan,
            retries=getattr(settings, "AGENT_RETRIES", 2),
        )

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _get_cached_plan(self, cache_key: str) -> QueryPlan | None:
        """Retrieve cached plan if still valid."""
        if cache_key in _PLAN_CACHE:
            plan, timestamp = _PLAN_CACHE[cache_key]
            if time.time() - timestamp < _PLAN_CACHE_TTL:
                return plan
            else:
                del _PLAN_CACHE[cache_key]
        return None

    def _cache_plan(self, cache_key: str, plan: QueryPlan) -> None:
        """Cache a successful plan with size limit."""
        # Evict oldest entries if cache is full
        if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX_SIZE:
            oldest_key = min(_PLAN_CACHE.keys(), key=lambda k: _PLAN_CACHE[k][1])
            del _PLAN_CACHE[oldest_key]

        _PLAN_CACHE[cache_key] = (plan, time.time())

    async def plan(self, query: str) -> QueryPlan:
        """Generate a query plan using LLM intelligence.

        Uses caching to reduce API calls for similar queries.

        Args:
            query: User's natural language query.

        Returns:
            QueryPlan with methods, reasoning, and entities.
        """
        self._initialize_agent()

        # Check cache first
        cache_key = self._get_cache_key(query)
        cached = self._get_cached_plan(cache_key)
        if cached is not None:
            logger.debug(f"Using cached query plan for: {query[:50]}...")
            return cached

        logger.info(f"Planning query execution for: {query[:50]}...")

        try:
            result = await self.agent.run(query)
            plan = result.output

            # Validate plan has at least one method
            if not plan.methods:
                plan.methods = [QueryMethod.SEMANTIC_SEARCH]

            # Ensure algorithm is set when graph_algorithms is in methods
            if QueryMethod.GRAPH_ALGORITHMS in plan.methods and plan.algorithm == GraphAlgorithm.NONE:
                plan.algorithm = GraphAlgorithm.SIMILARITY  # Safe default

            # Cache successful result
            self._cache_plan(cache_key, plan)

            logger.info(
                f"Query plan: methods={[m.value for m in plan.methods]}, "
                f"intent={plan.intent}, entities={plan.expected_entities[:3]}"
            )
            return plan

        except Exception as e:
            logger.warning(f"LLM query planning failed: {e}. Using fallback plan.")
            # Return a safe default plan
            return QueryPlan(
                methods=[QueryMethod.SEMANTIC_SEARCH],
                reasoning="Fallback plan due to LLM planning failure",
                fallback_methods=[QueryMethod.KEYWORD_SEARCH],
                expected_entities=[],
                requires_file_read=False,
                intent=None,
                algorithm=GraphAlgorithm.NONE,
            )

    def clear_cache(self) -> None:
        """Clear the plan cache."""
        _PLAN_CACHE.clear()
```

**Update:** `codebase_rag/retrieval/query_orchestrator.py`

Replace the intent classification + method map logic with:

```python
from codebase_rag.orchestrator.llm_query_planner import LLMQueryPlanner, QueryPlan


async def execute_async(
    self,
    query: str,
    top_k: int = 5,
    max_methods: int = 3,
) -> CombinedQueryResult:
    start = time.time()

    # LLM-driven planning replaces keyword-based intent classification
    planner = LLMQueryPlanner()
    plan = await planner.plan(query)

    methods_to_run = plan.methods[:max_methods]
    if not methods_to_run:
        methods_to_run = [QueryMethod.SEMANTIC_SEARCH]

    results = await self._execute_methods_async(query, methods_to_run, top_k, plan)

    # If primary methods returned empty, try fallbacks
    if not any(r.items for r in results) and plan.fallback_methods:
        logger.info(f"Primary methods returned empty, trying fallbacks: {plan.fallback_methods[:2]}")
        fallback_results = await self._execute_methods_async(
            query, plan.fallback_methods[:2], top_k, plan
        )
        results.extend(fallback_results)

    # Final fallback: semantic search if everything else failed
    if not any(r.items for r in results):
        logger.info("All methods returned empty, falling back to semantic search")
        semantic_result = await self._execute_semantic_search(query, top_k, start)
        results.append(semantic_result)

    return self._combine_results(query, plan, results, start)
```

#### 1.2 Replace Template-Based Cypher Fallback

**Remove:** `_try_template_fallback()` from `query_orchestrator.py`.

**Replace with:** LLM-driven Cypher repair/fallback with caching.

**Update:** `codebase_rag/services/llm.py` — add to `CypherGenerator`:

```python
# Cache for validated Cypher queries (hash -> (is_safe, timestamp))
_CYPHER_VALIDATION_CACHE: dict[str, tuple[bool, float]] = {}
_CYPHER_VALIDATION_CACHE_TTL = 3600  # 1 hour


class CypherGenerator:
    __slots__ = ("agent", "_fallback_agent")

    def __init__(self) -> None:
        # ... existing init ...
        self._fallback_agent = None

    async def generate_fallback(
        self,
        user_query: str,
        error_message: str,
    ) -> str | None:
        """Generate a conservative fallback Cypher query when primary generation fails.

        Args:
            user_query: Original natural language query.
            error_message: Error from the primary Cypher generation attempt.

        Returns:
            A conservative Cypher query string, or None if generation fails.
        """
        if self._fallback_agent is None:
            # Lazy init fallback agent with simpler system prompt
            from pydantic_ai import Agent
            self._fallback_agent = Agent(
                model=self.agent.model,
                system_prompt=(
                    "You generate conservative, simple Cypher queries. "
                    "Use only MATCH, WHERE, RETURN, LIMIT, ORDER BY. "
                    "No CALL procedures, no complex patterns. "
                    "Return ONLY the Cypher query, no markdown, no explanation."
                ),
                retries=1,
            )

        fallback_prompt = f"""The primary Cypher generation failed with this error: {error_message}

User query: {user_query}

Generate a SIMPLE, conservative Cypher query that will likely work. Guidelines:
- Use only MATCH, WHERE, RETURN, LIMIT, ORDER BY
- Do not use CALL procedures
- Do not use variable-length paths longer than 3 hops
- Prefer exact name/qualified_name matching over complex patterns
- If unsure, return a query that matches nodes by name CONTAINS and returns their qualified_name and path

Return ONLY the Cypher query string, no markdown, no explanation."""

        try:
            result = await self._fallback_agent.run(fallback_prompt)
            query = str(result.output).strip()
            # Remove markdown code blocks if present
            if query.startswith("```"):
                query = query.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return query
        except Exception as e:
            logger.warning(f"Fallback Cypher generation failed: {e}")
            return None
```

**Update:** `codebase_rag/retrieval/query_orchestrator.py`

In `_execute_graph_traversal_async()`, update the fallback chain:

```python
async def _execute_graph_traversal_async(
    self,
    query: str,
    top_k: int,
    plan: QueryPlan,
    start: float,
) -> QueryResult:
    """Execute graph traversal with LLM fallback chain."""
    cypher_gen = CypherGenerator()
    cypher: str | None = None
    last_error: str = "unknown error"

    # Step 1: LLM Cypher generation
    try:
        cypher = await cypher_gen.generate(query)
        if cypher:
            results = await self._fetch_all_async(cypher)
            if results:
                return self._build_success_result(query, results, start, "graph_traversal")
    except Exception as e:
        last_error = str(e)
        logger.warning(f"Primary Cypher generation failed: {e}")

    # Step 1b: If syntax error, try LLM repair
    if cypher:
        try:
            cypher = await cypher_gen.repair(query, cypher, last_error)
            if cypher:
                results = await self._fetch_all_async(cypher)
                if results:
                    return self._build_success_result(query, results, start, "graph_traversal_repaired")
        except Exception as repair_error:
            last_error = str(repair_error)
            logger.warning(f"Cypher repair failed: {repair_error}")

    # Step 2: LLM fallback — generate simpler conservative query
    try:
        fallback_cypher = await cypher_gen.generate_fallback(query, last_error)
        if fallback_cypher:
            results = await self._fetch_all_async(fallback_cypher)
            if results:
                return self._build_success_result(query, results, start, "graph_traversal_fallback")
    except Exception as fallback_error:
        logger.warning(f"Fallback Cypher failed: {fallback_error}")

    # Step 3: Use expected_entities from plan for keyword-based lookup
    if plan and plan.expected_entities:
        entity = plan.expected_entities[0]
        keyword_result = await self._execute_keyword_search(entity, top_k, start)
        if keyword_result.items:
            return keyword_result

    # Step 4: Semantic fallback (last resort)
    return QueryResult(
        query=query,
        items=[],
        method_used="graph_traversal_failed",
        elapsed_time=time.time() - start,
        error="All graph traversal attempts failed",
    )
```

**Helper:** `_build_success_result` constructs a `QueryResult` from raw graph rows:

```python
def _build_success_result(
    self,
    query: str,
    items: list[dict[str, object]],
    start: float,
    method_used: str,
) -> QueryResult:
    """Build a successful QueryResult from raw items."""
    return QueryResult(
        query=query,
        items=items,
        method_used=method_used,
        elapsed_time=time.time() - start,
    )
```

#### 1.3 Replace Graph Algorithm Selection

**Remove:** Keyword-based algorithm selection in `_execute_graph_algorithms_async()`.

**Replace with:** Algorithm selection via LLM planner (`QueryPlan` already includes `graph_algorithms` when appropriate). The graph algorithm tool accepts parameters that the LLM sets directly based on the plan's reasoning.

**Update:** `codebase_rag/retrieval/query_orchestrator.py`

```python
from codebase_rag.orchestrator.llm_query_planner import GraphAlgorithm

async def _execute_graph_algorithms_async(
    self,
    query: str,
    top_k: int,
    plan: QueryPlan,
    start: float,
) -> QueryResult:
    """Execute graph algorithms based on LLM plan.

    Algorithm selection is mechanical: the LLM outputs plan.algorithm
    as a structured enum, and Python executes the corresponding method.
    No keyword matching on reasoning text.
    """
    if not self._algorithms:
        return QueryResult(query=query, items=[], method_used="graph_algorithms_unavailable", elapsed_time=0)

    entities = plan.expected_entities if plan else []
    algorithm = plan.algorithm if plan else GraphAlgorithm.NONE

    try:
        match algorithm:
            case GraphAlgorithm.SHORTEST_PATH:
                if len(entities) >= 2:
                    result = await asyncio.to_thread(
                        self._algorithms.find_shortest_path,
                        entities[0],
                        entities[1],
                    )
                    return self._build_success_result(query, [result], start, "graph_algorithm_path")

            case GraphAlgorithm.SIMILARITY:
                if entities:
                    results = await asyncio.to_thread(
                        self._algorithms.get_similar_nodes,
                        entities[0],
                        top_k,
                    )
                    return self._build_success_result(query, results, start, "graph_algorithm_similarity")

            case GraphAlgorithm.COMMUNITY:
                results = await asyncio.to_thread(
                    self._algorithms.detect_communities,
                )
                return self._build_success_result(query, results, start, "graph_algorithm_community")

            case GraphAlgorithm.BFS:
                if entities:
                    results = await asyncio.to_thread(
                        self._algorithms.get_bfs_context,
                        entities[0],
                        3,
                    )
                    return self._build_success_result(query, results, start, "graph_algorithm_bfs")

            case GraphAlgorithm.NONE | _:
                # Default: similarity search using first entity
                if entities:
                    results = await asyncio.to_thread(
                        self._algorithms.get_similar_nodes,
                        entities[0],
                        top_k,
                    )
                    return self._build_success_result(query, results, start, "graph_algorithm_similarity_default")

    except Exception as e:
        logger.error(f"Graph algorithm execution failed: {e}")

    return QueryResult(
        query=query,
        items=[],
        method_used="graph_algorithms_failed",
        elapsed_time=time.time() - start,
    )
```

---

### Part 2: LLM-Driven Eligibility & Safety

#### 2.1 Remove Regex-Based Eligibility Classifier

**File:** `codebase_rag/orchestrator/concurrency_eligibility_classifier.py`

**Remove:**
- `EXPLICIT_PARALLEL_PATTERNS`
- `EXPLICIT_SEQUENTIAL_PATTERNS`
- `SAFETY_NON_ELIGIBLE_PATTERNS`
- `_is_conceptual_question()` (regex-based version)

**Keep:** `_get_llm_eligibility()` — this is already the correct approach.

**Update:** `is_eligible()` to remove all regex gates:

```python
from codebase_rag.orchestrator.llm_query_planner import LLMQueryPlanner


class ConcurrencyEligibilityClassifier:
    """Classifies tasks for parallel execution eligibility using LLM intelligence."""

    __slots__ = (
        "enabled",
        "threshold",
        "_effective_threshold",
        "min_subtask_count",
        "agent",
        "success_rate_tracker",
        "adaptive_adjustment_enabled",
        "_codebase_context_cache",
    )

    LLM_ELIGIBILITY_PROMPT = """You are a parallel task eligibility classifier for a codebase analysis system.
Evaluate if the user request (provided separately) can be split into independent,
non-overlapping subtasks that can be executed in parallel.

Codebase Context:
- Total files: {file_count}
- Primary languages: {languages}
- Repository size: {repo_size}
- Query mode: {query_mode}

Respond ONLY with a valid JSON object:
{{
  "eligible": boolean,
  "confidence": float between 0.0 and 1.0,
  "task_type": string,
  "reasoning": string,
  "fallback_action": string or null
}}

Safety Rules:
- NEVER parallelize tasks that modify, create, delete, or update files/code/config
- ALWAYS parallelize read-only tasks that analyze multiple files or entities
- For conceptual/theoretical questions, return eligible=false with task_type="conceptual_question"
- For explicit write requests ("fix this", "create a file", "refactor X"), return eligible=false with task_type="write_operation"
- When uncertain, prefer sequential execution (eligible=false)
"""

    def __init__(self) -> None:
        self.enabled: bool = getattr(settings, "CGR_AUTO_PARALLEL_ENABLED", True)
        self.threshold: float = getattr(settings, "CGR_PARALLEL_ELIGIBILITY_THRESHOLD", 0.6)
        self._effective_threshold: float = self.threshold
        self.min_subtask_count: int = getattr(settings, "CGR_PARALLEL_MIN_SUBTASKS", 2)
        self.agent = None
        self.success_rate_tracker: dict[str, list[bool]] = {}
        self.adaptive_adjustment_enabled: bool = getattr(settings, "CGR_PARALLEL_ADAPTIVE_THRESHOLD", True)
        self._codebase_context_cache: dict[str, object] | None = None
        self._planner: LLMQueryPlanner | None = None

    async def is_eligible(
        self,
        prompt: str,
        subtask_count: int | None = None,
        has_write_operations: bool = False,
        query_mode: QueryMode = QueryMode.CODE_ONLY,
    ) -> EligibilityResult:
        """Determine if a task is eligible for automatic parallel execution.

        All semantic decisions are delegated to the LLM. No regex pre-filtering.
        """
        if not self.enabled:
            return EligibilityResult(False, "concurrency_disabled", 0.0)

        if has_write_operations:
            return EligibilityResult(False, "write_operation", 0.0)

        if subtask_count is not None and subtask_count < self.min_subtask_count:
            return EligibilityResult(False, "insufficient_subtasks", 0.0)

        # Delegate ALL semantic decisions to the LLM
        confidence, task_type = await self._get_llm_eligibility(prompt, query_mode)
        effective_threshold = self._adjust_threshold_based_on_success(task_type)

        if confidence >= effective_threshold:
            return EligibilityResult(True, task_type, confidence)

        return EligibilityResult(False, task_type, confidence)

    async def _get_llm_eligibility(
        self,
        prompt: str,
        query_mode: QueryMode,
    ) -> tuple[float, str]:
        """Use LLM to determine eligibility.

        Passes query_mode to the system prompt for context-aware classification.
        """
        if not self.agent:
            config = settings.active_orchestrator_config
            provider = get_provider_from_config(config)
            llm = provider.create_model(config.model_id)

            if getattr(settings, "CGR_PARALLEL_CODEBASE_CONTEXT", True):
                file_count, languages, repo_size = self._get_codebase_context()
            else:
                file_count = 0
                languages = "unknown"
                repo_size = "unknown"

            system_prompt = self.LLM_ELIGIBILITY_PROMPT.format(
                file_count=file_count,
                languages=languages,
                repo_size=repo_size,
                query_mode=query_mode.value,
            )

            self.agent = Agent(
                model=llm,
                system_prompt=system_prompt,
                output_type=dict,
                retries=settings.AGENT_RETRIES,
            )

        try:
            result = await self.agent.run(
                prompt, usage_limits=UsageLimits(request_limit=settings.AGENT_REQUEST_LIMIT)
            )
            result_data_raw = result.output
            if not isinstance(result_data_raw, dict):
                return 0.0, "llm_invalid_output"

            result_data = result_data_raw
            confidence = max(
                0.0,
                min(1.0, self._coerce_float(result_data.get("confidence", 0.0))),
            )
            task_type = (
                result_data.get("task_type", "llm_analyzed")
                if result_data.get("eligible", False)
                else "llm_rejected"
            )
            reasoning = result_data.get("reasoning", "")
            if reasoning:
                logger.debug(f"LLM eligibility reasoning: {reasoning}")

            return confidence, str(task_type)
        except Exception as e:
            logger.warning(f"LLM eligibility check failed: {e}")
            return 0.0, "llm_check_failed"
```

#### 2.2 Replace Write Intent Detection

**File:** `codebase_rag/main.py:1717-1823`

**Remove:** `_has_write_intent()` entirely.

**Replace with:** The LLM eligibility classifier handles write detection via its system prompt. No separate Python function needed.

**Update:** Call sites that use `_has_write_intent()` must be updated.

In `main.py` (around line 2162), change:

```python
# BEFORE:
has_write_operations = _has_write_intent(question_with_context)
# ... later ...
eligibility = await classifier.is_eligible(
    prompt=question_with_context,
    has_write_operations=has_write_operations,
    ...
)

# AFTER:
# Let the LLM classifier detect write intent via structured analysis.
# The classifier's system prompt explicitly instructs it to reject
# write operations with eligible=false and task_type="write_operation".
eligibility = await classifier.is_eligible(
    prompt=question_with_context,
    has_write_operations=False,
    ...
)
```

**Rationale:** The 4-layer regex in `_has_write_intent()` produces false positives
("explain what the write method does" → blocked) and false negatives
("make sure the config is updated" → not caught by Layer 1). The LLM classifier
has full context and explicit safety rules, making it strictly more reliable.

---

### Part 3: LLM-Driven Task Splitting

#### 3.1 Replace Strategy Detection

**File:** `codebase_rag/orchestrator/task_splitter.py`

**Remove:** `_detect_strategy()` with keyword indicators.

**Replace with:** LLM strategy planner using `LLMQueryPlanner`.

**Update:** `TaskSplitter` class:

```python
from codebase_rag.orchestrator.llm_query_planner import LLMQueryPlanner, QueryPlan


class SplitStrategy(BaseModel):
    """Structured output for task splitting decisions."""

    strategy: str = Field(
        ...,
        description="One of: file, node, query, sequential",
    )
    reasoning: str = Field(
        ...,
        description="Explanation for strategy selection",
    )
    relevant_extensions: list[str] = Field(
        default_factory=list,
        description="File extensions to include (e.g., ['.py', '.js'])",
    )
    relevant_name_patterns: list[str] = Field(
        default_factory=list,
        description="Name patterns to match (e.g., ['_test', 'test_'])",
    )
    relevant_paths: list[str] = Field(
        default_factory=list,
        description="Directory paths to scope the search",
    )
    complexity: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Task complexity 1-5 (affects subtask limits)",
    )
    max_subtasks: int | None = Field(
        default=None,
        description="Maximum subtasks to create",
    )


class TaskSplitter:
    """Splits tasks for parallel execution using LLM intelligence."""

    __slots__ = (
        "repo_path",
        "planner",
        "_extension_hints",
        "_name_pattern_hints",
        "_path_hints",
        "_strategy_agent",
    )

    STRATEGY_SYSTEM_PROMPT = """You are a task splitting strategist. Given a user request, determine:
1. How should this task be parallelized?
2. What files or entities are relevant?

Available strategies:
- file: One subtask per relevant file
- node: One subtask per relevant node type (functions, classes, etc.)
- query: One subtask per semantic sub-question
- sequential: Do not parallelize

Return JSON with your analysis."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        self.planner = LLMQueryPlanner()
        self._extension_hints: list[str] = []
        self._name_pattern_hints: list[str] = []
        self._path_hints: list[str] = []
        self._strategy_agent = None

    def _get_strategy_agent(self):
        """Lazy initialization of strategy agent."""
        if self._strategy_agent is None:
            from pydantic_ai import Agent
            from codebase_rag.providers import _create_provider_model

            config = settings.active_orchestrator_config  # Reuse config
            llm = _create_provider_model(config)

            self._strategy_agent = Agent(
                model=llm,
                system_prompt=self.STRATEGY_SYSTEM_PROMPT,
                output_type=SplitStrategy,
                retries=1,
            )
        return self._strategy_agent

    async def _plan_split_strategy(self, prompt: str) -> SplitStrategy:
        """Use LLM to determine splitting strategy and scope."""
        agent = self._get_strategy_agent()
        try:
            result = await agent.run(prompt)
            return result.output
        except Exception as e:
            logger.warning(f"LLM strategy planning failed: {e}. Using file strategy.")
            return SplitStrategy(
                strategy="file",
                reasoning="Fallback due to LLM failure",
                relevant_extensions=[],
                relevant_name_patterns=[],
                relevant_paths=[],
                complexity=2,
            )

    async def split_task(
        self,
        prompt: str,
        strategy: str = "auto",
        max_subtasks: int | None = None,
    ) -> list[Subtask]:
        """Split a task into subtasks using LLM intelligence.

        Precedence for max_subtasks:
        1. Explicit argument to split_task() (highest priority)
        2. LLM-planned SplitStrategy.max_subtasks (if strategy is auto)
        3. No limit (None)
        """
        if strategy == "auto":
            plan = await self._plan_split_strategy(prompt)
            strategy = plan.strategy
            # Explicit argument overrides LLM plan; LLM plan overrides None
            max_subtasks = max_subtasks if max_subtasks is not None else plan.max_subtasks
            self._extension_hints = plan.relevant_extensions
            self._name_pattern_hints = plan.relevant_name_patterns
            self._path_hints = plan.relevant_paths

        # ... existing strategy dispatch ...
```

#### 3.2 Remove Keyword-Based Complexity Scoring

**File:** `codebase_rag/orchestrator/task_splitter.py`

**Remove:** The `simple_keywords` / `complex_keywords` block.

**Replace with:** Use `complexity` from the LLM planner output (1-5 scale with reasoning).

#### 3.3 Remove Keyword-Based File Type Hints

**File:** `codebase_rag/orchestrator/task_splitter.py`

**Remove:** The entire keyword-based extension and name pattern hint extraction.

**Replace with:** Use `relevant_extensions` and `relevant_name_patterns` from the `SplitStrategy` output.

#### 3.4 Remove Regex-Based Scope Path Extraction

**File:** `codebase_rag/orchestrator/task_splitter.py`

**Remove:** The `re.findall()` path extraction.

**Replace with:** Use `relevant_paths` from the `SplitStrategy` output.

---

### Part 4: LLM-Driven Sufficiency Analysis

**File:** `codebase_rag/orchestrator/sufficiency_analyzer.py`

**Remove:** `_DIAGNOSTIC_KEYWORDS`, `_FUNCTIONAL_KEYWORDS`, `_STRUCTURAL_KEYWORDS`, and the `analyze_requirements()` function.

**Replace with:** LLM-driven sufficiency analyzer.

**New file:** `codebase_rag/orchestrator/llm_sufficiency_analyzer.py`

```python
from __future__ import annotations

from pydantic import BaseModel, Field
from __future__ import annotations

from loguru import logger

from codebase_rag.config import settings


class SufficiencyAssessment(BaseModel):
    """LLM assessment of whether current results answer the user's question."""

    sufficient: bool = Field(
        ...,
        description="Whether the results fully answer the question",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of the assessment",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="What information is still needed",
    )
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="Tools that might fill gaps",
    )
    requires_cross_validation: bool = Field(
        default=False,
        description="Whether code-doc cross-validation is needed",
    )


class LLMSufficiencyAnalyzer:
    """Uses LLM to determine if query results are sufficient."""

    __slots__ = ("agent",)

    SYSTEM_PROMPT = """You are a sufficiency analyzer. Given a user's question and the current results from various tools, determine if the answer is complete.

Respond with JSON:
{
  "sufficient": boolean,
  "reasoning": "...",
  "missing_information": ["what else is needed"],
  "suggested_tools": ["tool_name", ...],
  "requires_cross_validation": boolean
}

Rules:
- If results directly answer the question: sufficient=true
- If results are partial or ambiguous: sufficient=false with missing_information
- If the question requires verifying code against docs: requires_cross_validation=true
- suggested_tools should list tools that might fill gaps (e.g., "read_file", "semantic_search", "get_call_hierarchy")"""

    def __init__(self) -> None:
        self.agent = None

    def _initialize_agent(self) -> None:
        """Lazy initialization."""
        if self.agent is None:
            from pydantic_ai import Agent
            from codebase_rag.providers import _create_provider_model

            config = settings.active_orchestrator_config
            llm = _create_provider_model(config)

            self.agent = Agent(
                model=llm,
                system_prompt=self.SYSTEM_PROMPT,
                output_type=SufficiencyAssessment,
                retries=getattr(settings, "AGENT_RETRIES", 1),
            )

    async def assess(
        self,
        question: str,
        results: list[dict[str, object]],
    ) -> SufficiencyAssessment:
        """Assess whether results are sufficient to answer the question."""
        self._initialize_agent()

        context = f"Question: {question}\n\nResults: {results}"
        try:
            result = await self.agent.run(context)
            return result.output
        except Exception as e:
            logger.warning(f"Sufficiency assessment failed: {e}")
            return SufficiencyAssessment(
                sufficient=False,
                reasoning=f"Assessment failed: {e}",
                missing_information=["Unknown - assessment failed"],
                suggested_tools=[],
            )
```

---

### Part 5: Tool Unfiltering (Mode as Hint)

**File:** `codebase_rag/tools/__init__.py`

**Remove:** `_CODE_ONLY_TOOL_NAMES`, `_DOCUMENT_ONLY_TOOL_NAMES`, `_BOTH_MERGED_TOOL_NAMES`, and `get_tools_for_mode()`.

**Replace with:** Mode-aware tool descriptions that guide the LLM without hiding tools.

**Update:** Tool descriptions to include mode hints:

```python
# In codebase_rag/tools/tool_descriptions.py (or inline)

QUERY_DOCUMENT_GRAPH = (
    "Query the DOCUMENT graph/vector. "
    "Use for questions about documentation, tutorials, guides, or API docs. "
    "When the user's question is about code implementation, prefer query_graph or semantic_search instead. "
    "Examples: 'How do I use the authentication API?', 'What does the docs say about configuration?'"
)

QUERY_CODE_GRAPH = (
    "Query the CODE knowledge graph using natural language. "
    "Use for questions about classes, functions, methods, dependencies, or code structure. "
    "When the user's question is about documentation, prefer query_document_graph instead. "
    "Examples: 'Find all functions that call each other', 'What classes are in the user module'"
)
```

**Update:** The agent initialization to pass ALL tools regardless of mode. Include mode in the system prompt as context:

```python
def _build_system_prompt(mode: QueryMode) -> str:
    """Build system prompt with mode context (not filtering)."""
    return f"""You are a codebase analysis assistant. Current query mode: {mode.value}.

Mode descriptions:
- code_only: Focus on code graph queries. Document tools are available but secondary.
- document_only: Focus on document graph queries. Code tools are available but secondary.
- both_merged: Query both graphs and merge results.
- code_vs_doc: Validate code against documentation.
- doc_vs_code: Validate documentation against code.

Use your judgment to select the most appropriate tools for the user's query.
All tools are available to you. Select based on what the user is actually asking."""
```

**CRITICAL:** `QueryRouter` currently enforces hard cross-mode boundaries
(e.g., `_query_document_only()` returns an error when `current_mode == CODE_ONLY`).
Before unfiltering tools, update `QueryRouter` to allow cross-mode execution
when the request comes from the LLM agent. Add a `forced: bool = False` flag
to `QueryRequest` that bypasses mode guards:

```python
@dataclass
class QueryRequest:
    ...
    forced: bool = False  # If True, bypass current_mode guards
```

In `QueryRouter._query_document_only()` and `_query_code_only()`, change:

```python
# From:
if self.current_mode == QueryMode.CODE_ONLY:
    return QueryResponse(answer="Document queries are disabled...", ...)

# To:
if self.current_mode == QueryMode.CODE_ONLY and not request.forced:
    return QueryResponse(answer="Document queries are disabled...", ...)
```

The agent sets `forced=True` when it explicitly chooses a cross-mode tool.
This preserves backward compatibility for direct API users while allowing
the LLM full tool access.

---

### Part 6: Query Mode Selection

**File:** `codebase_rag/main.py`

**Update:** `_determine_default_query_mode()` keeps existing behavior for backward
compatibility, but adds a configuration flag for users who want the new LLM-First default.

**Recommended approach:**

```python
async def _infer_query_mode_from_query(query: str) -> QueryMode:
    """Optionally use LLM to infer mode from first query (can be skipped for performance)."""
    from codebase_rag.orchestrator.llm_query_planner import LLMQueryPlanner

    planner = LLMQueryPlanner()
    plan = await planner.plan(query)

    if plan.intent:
        intent = plan.intent
        if intent in ("doc_semantic_search", "doc_graph_traversal", "doc_procedural", "doc_comparison"):
            return QueryMode.DOCUMENT_ONLY
        if intent == "validation":
            return QueryMode.CODE_VS_DOC

    return QueryMode.CODE_ONLY  # Safe default


def _determine_default_query_mode(
    code_graph,
    doc_graph,
    user_query: str | None = None,
) -> QueryMode:
    """Determine query mode based on available graphs.

    Mode is a hint for the LLM agent, not a tool filter.
    """
    code_count, doc_count = _get_content_availability(code_graph, doc_graph)

    # If only one graph has content, use that mode
    if code_count == 0 and doc_count > 0:
        return QueryMode.DOCUMENT_ONLY
    if code_count > 0 and doc_count == 0:
        return QueryMode.CODE_ONLY

    # Both graphs have content
    # Backward compatible default: BOTH_MERGED (existing behavior)
    # New opt-in default: CODE_ONLY (set CGR_DEFAULT_MODE_BOTH="code_only")
    default_when_both = getattr(settings, "CGR_DEFAULT_MODE_WHEN_BOTH", "both_merged")
    if default_when_both == "code_only":
        return QueryMode.CODE_ONLY
    return QueryMode.BOTH_MERGED
```

**Configuration:**

```python
# codebase_rag/config.py
CGR_DEFAULT_MODE_WHEN_BOTH: str = Field(default="both_merged", pattern="^(both_merged|code_only)$")
```

This preserves existing user experience while allowing opt-in to the simpler default.
The LLM agent always sees all tools, so even in `CODE_ONLY` mode it can access
document tools via `forced=True` requests.

---

### Part 7: Cypher Validation Hybrid

**File:** `codebase_rag/services/llm.py`

**Current approach:** Pure regex blacklist for Cypher validation.

**Problem:** Regex cannot understand Cypher semantics. It produces false positives (blocking safe queries) and false negatives (allowing unsafe ones).

**Proposed hybrid approach:**

1. **Fast-path mechanical checks (zero false positives):** Continue blocking obvious dangers:
   - Multiple semicolons (`;` count > 1) → injection risk
   - `OVER(` → unsupported feature
   - `USING PARALLEL EXECUTION` → unsupported

2. **Semantic validation via LLM with caching:** Send the query to LLM for semantic safety check, caching results.

**Update:** `_validate_cypher_read_only()`

```python
# Cache for LLM-validated Cypher queries
_CYPHER_LLM_VALIDATION_CACHE: dict[str, tuple[bool, float]] = {}
_CYPHER_LLM_VALIDATION_CACHE_TTL = 3600  # 1 hour


async def _validate_cypher_read_only(query: str) -> None:
    """Validate Cypher query for read-only safety.

    Uses hybrid approach: mechanical fast-path + LLM semantic validation.
    """
    upper_query = query.upper()

    # Fast-path mechanical checks (zero false positives)
    if upper_query.count(";") > 1:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="Multiple semicolon-separated queries",
                query=query,
            )
        )
    if "OVER(" in upper_query:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="OVER() window function",
                query=query,
            )
        )
    if "USING PARALLEL EXECUTION" in upper_query:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="USING PARALLEL EXECUTION",
                query=query,
            )
        )

    # Check cache for LLM validation
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    if query_hash in _CYPHER_LLM_VALIDATION_CACHE:
        is_safe, _ = _CYPHER_LLM_VALIDATION_CACHE[query_hash]
        if is_safe:
            return
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="Unsafe operation (cached)",
                query=query,
            )
        )

    # LLM semantic validation
    is_safe = await _llm_validate_cypher_safety(query)

    # Cache result
    _CYPHER_LLM_VALIDATION_CACHE[query_hash] = (is_safe, time.time())

    if not is_safe:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="Unsafe write operation detected by LLM",
                query=query,
            )
        )


class CypherSafetyAssessment(BaseModel):
    """Structured output for Cypher safety validation."""

    safe: bool = Field(..., description="Whether the query is read-only safe")
    reasoning: str = Field(default="", description="Brief explanation")


# Singleton agent for Cypher safety validation
_CYPHER_SAFETY_AGENT: Agent | None = None


async def _llm_validate_cypher_safety(query: str) -> bool:
    """Use LLM to validate Cypher query safety semantically.

    Uses structured output (Pydantic model) instead of string parsing.
    Agent is created once and reused for efficiency.
    """
    global _CYPHER_SAFETY_AGENT

    if _CYPHER_SAFETY_AGENT is None:
        from pydantic_ai import Agent

        system_prompt = """You validate Cypher queries for read-only safety. A query is UNSAFE if it:
- Modifies data (CREATE, DELETE, DETACH DELETE, SET, REMOVE, MERGE that creates new nodes)
- Modifies schema (CREATE INDEX, DROP INDEX, CREATE CONSTRAINT, etc.)
- Uses non-read-only CALL procedures (write procedures, admin procedures)
- Contains multiple statements separated by semicolons

A query is SAFE if it only:
- Reads data (MATCH, RETURN, WITH, OPTIONAL MATCH, UNWIND)
- Uses read-only CALL procedures
- Uses COUNT, COLLECT, etc.

Respond with a JSON object: {"safe": true/false, "reasoning": "brief explanation"}"""

        config = settings.active_cypher_config
        llm = _create_provider_model(config)
        _CYPHER_SAFETY_AGENT = Agent(
            model=llm,
            system_prompt=system_prompt,
            output_type=CypherSafetyAssessment,
            retries=1,
        )

    try:
        result = await _CYPHER_SAFETY_AGENT.run(query)
        return result.output.safe
    except Exception as e:
        logger.warning(f"LLM Cypher validation failed: {e}. Defaulting to unsafe.")
        return False
```

**Note:** Remove `CYPHER_DANGEROUS_KEYWORDS` and `CYPHER_SAFE_CALL_PROCEDURES` from `constants.py` after this is implemented.

---

### Part 8: Remove Keyword Extraction Utilities

**File:** `codebase_rag/utils/query_utils.py`

**Remove:** `extract_best_keyword()` and `extract_keywords()`.

**Rationale:** These were used for:
1. Template fallback Cypher queries (removed in Part 1.2)
2. Keyword search fallback (should use LLM-extracted entities from `QueryPlan.expected_entities`)

**Replace with:** Use `expected_entities` from `LLMQueryPlanner.plan()` for name-based lookups.

---

## Implementation Plan

### Phase 1: Core Orchestration (Week 1-2)

| Task | Files | Priority |
|------|-------|----------|
| Add orchestrator config to settings | `config.py` | P0 |
| Implement `LLMQueryPlanner` with Pydantic output | `orchestrator/llm_query_planner.py` (new) | P0 |
| Replace `_INTENT_KEYWORD_MAP` + `_INTENT_METHOD_MAP` with planner | `retrieval/query_orchestrator.py` | P0 |
| Remove `_try_template_fallback()` and add LLM fallback to `CypherGenerator` | `retrieval/query_orchestrator.py`, `services/llm.py` | P0 |
| Remove keyword-based graph algorithm selection | `retrieval/query_orchestrator.py` | P0 |
| Remove `extract_best_keyword()` / `extract_keywords()` | `utils/query_utils.py` | P1 |
| Add tests for LLM query planner | `tests/test_llm_query_planner.py` (new) | P1 |
| Add tests for fallback chains | `tests/test_query_orchestrator.py` | P1 |

**Test requirements for fallback chains:**
- Mock `LLMQueryPlanner` to return empty `methods` → assert fallback to semantic search.
- Mock planner failure → assert safe default plan is used.
- Mock empty primary results + non-empty fallback methods → assert fallback results are included.
- Mock `GraphAlgorithm.SHORTEST_PATH` with < 2 entities → assert fallback to similarity/default.

### Phase 2: Eligibility & Safety (Week 2-3)

| Task | Files | Priority |
|------|-------|----------|
| Remove all regex patterns from `ConcurrencyEligibilityClassifier` | `orchestrator/concurrency_eligibility_classifier.py` | P0 |
| Enhance `LLM_ELIGIBILITY_PROMPT` with mode awareness | `orchestrator/concurrency_eligibility_classifier.py` | P0 |
| Remove `_has_write_intent()` from `main.py` | `main.py` | P0 |
| Implement hybrid Cypher validation (mechanical + LLM) | `services/llm.py` | P1 |
| Remove `CYPHER_DANGEROUS_KEYWORDS` and `CYPHER_SAFE_CALL_PROCEDURES` | `constants.py` | P2 |
| Add tests for LLM eligibility classifier | `tests/test_concurrency_eligibility.py` | P1 |

### Phase 3: Task Splitting (Week 3-4)

| Task | Files | Priority |
|------|-------|----------|
| Implement `_plan_split_strategy()` with LLM | `orchestrator/task_splitter.py` | P0 |
| Remove `_detect_strategy()` keyword matching | `orchestrator/task_splitter.py` | P0 |
| Remove keyword-based complexity scoring | `orchestrator/task_splitter.py` | P1 |
| Remove keyword-based file type hints | `orchestrator/task_splitter.py` | P1 |
| Remove regex-based scope path extraction | `orchestrator/task_splitter.py` | P1 |
| Add tests for LLM task splitting | `tests/test_task_splitter.py` | P1 |

### Phase 4: Sufficiency Analysis & Tool Unfiltering (Week 4-5)

| Task | Files | Priority |
|------|-------|----------|
| Implement `LLMSufficiencyAnalyzer` | `orchestrator/llm_sufficiency_analyzer.py` (new) | P1 |
| Remove keyword-based `analyze_requirements()` | `orchestrator/sufficiency_analyzer.py` | P1 |
| Remove `get_tools_for_mode()` hardcoded filtering | `tools/__init__.py` | P0 |
| Update tool descriptions with mode hints | `tools/tool_descriptions.py` | P0 |
| Update agent system prompt to include mode as context | `main.py` | P0 |
| Add tests for sufficiency analyzer | `tests/test_sufficiency_analyzer.py` (new) | P2 |

### Phase 5: Query Mode Selection (Week 5)

| Task | Files | Priority |
|------|-------|----------|
| Replace `_determine_default_query_mode()` with LLM inference or explicit default | `main.py` | P1 |
| Add mode inference tests | `tests/test_mode_selection.py` (new) | P2 |

---

## Backward Compatibility

1. **Configuration flags:** Add feature flags for gradual rollout:
   ```python
   CGR_LLM_FIRST_ORCHESTRATION: bool = True  # Master switch
   CGR_LLM_FIRST_ELIGIBILITY: bool = True
   CGR_LLM_FIRST_TASK_SPLITTING: bool = True
   CGR_LLM_FIRST_CYPHER_VALIDATION: bool = True
   ```

2. **Fallback chains:** When LLM planner fails (timeout, error), fall back to:
   - `QueryMethod.SEMANTIC_SEARCH` as default single method
   - Sequential execution for eligibility
   - `CODE_ONLY` mode for mode selection

3. **Existing specs:** The `document_graph_data_modeling_spec.md` uses `LLMQueryPlanner` for query intent classification (unified approach).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM planner latency adds overhead | Medium | Medium | Cache plans by query hash; use fast model for planning |
| LLM planner cost | Medium | Low | Planning is one call per query; cheaper than running wrong methods |
| LLM makes wrong plan | Low | Medium | Fallback methods + sufficiency analyzer catches gaps |
| Removed regex broke fast-path performance | Low | Low | Mechanical fast-paths (semicolon count, etc.) remain |
| LLM eligibility false positives for write ops | Low | High | Strong system prompt + user approval for write tools remains |
| Cache memory growth | Low | Low | LRU eviction with max size limits |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Intent classification accuracy | ~60% | >90% |
| False positive rate for eligibility | ~25% | <5% |
| Query response time (correct method first) | N/A | >80% |
| User satisfaction with tool selection | Low | High |
| Lines of rule-based classification code | ~400 | ~50 (only mechanical guards) |
