# Parallel Execution Optimization Fix Specification
**Version**: 2.0  
**Date**: 2025-01-19  
**Status**: Implementation-Ready  
**Author**: Code Graph RAG Team  
**Revision Notes**: v2.0 incorporates critical bug fixes and design corrections identified during codebase alignment review — see inline "Critical Bug Fix" and "Design Note" callouts in §1–§5 and §Config for details.  

## Executive Summary
This specification addresses the critical issue where parallel execution is frequently skipped due to overly conservative eligibility detection, insufficient subtask generation, and write operation false positives. The fix implements a balanced approach that maintains safety while significantly increasing parallel execution utilization through improved detection logic, enhanced task splitting, and intelligent write operation handling.

## Problem Analysis
### Current Issues Identified
1. **Overly Conservative Write Operation Detection**: The current `_has_write_intent()` function uses two broad regex patterns: `r"\b(create|write|edit|modify|update|delete|remove|refactor|rename|move|implement|fix|patch)\b"` which matches any write-related word **anywhere** in the prompt (not just as a verb acting on a code target), and `r"\badd\b.{0,40}\b(file|files|code|...)\b"` which catches "add file" patterns. The first pattern triggers false positives on legitimate read-only queries containing words like "write" or "create" in documentation contexts — e.g., "show me how the code creates objects" or "explain what the write method does" both falsely trigger write intent because `\b(create|write)\b` matches anywhere regardless of syntactic role.

2. **Insufficient Subtask Generation**: Task splitter's `_collect_scoped_files()` already falls back to returning all code files when no scope paths are found (`if not scope_paths: return all_files`). However, the real bottleneck is `_extract_scope_paths()` which fails to extract meaningful paths from many natural language prompts, causing the fallback to fire too often. When all files are returned, the resulting subtasks may be too numerous (thousands) or may include irrelevant files, leading to the task queue exceeding `CGR_PARALLEL_MAX_QUEUE_SIZE` and falling back to sequential execution. The actual issues are:
   - `_extract_scope_paths()` is overly aggressive in extracting candidates (it matches any path-like string via `r"(?:\.{0,2}/)?[A-Za-z0-9_./\\-]+"`), producing many false positives that fail `_normalize_candidate_path()` validation, resulting in no valid scope paths
   - No file-type-based narrowing when scope paths aren't found (e.g., "analyze all Python files" should filter to `.py` files rather than returning everything)
   - No prompt-language-based filtering to reduce subtask count and improve relevance

3. **Low LLM Confidence Scores**: The LLM eligibility classifier frequently returns confidence scores below the 0.7 threshold due to:
   - Inadequate prompt engineering for intent classification
   - Lack of context about available codebase structure
   - No feedback loop for confidence calibration

4. **Missing User Control**: Users cannot override automatic decisions when they know their task is safe for parallelization.

### Root Causes
- **Safety vs. Performance Trade-off**: Current implementation prioritizes safety over performance, erring on the side of sequential execution
- **Natural Language Understanding Gaps**: System struggles with diverse phrasing patterns in user requests
- **Lack of Adaptive Learning**: No mechanism to learn from successful parallel executions to improve future decisions

## Solution Design
### 1. Enhanced Write Operation Detection
**File**: `codebase_rag/main.py`

Replace the current `_has_write_intent()` function with a more sophisticated implementation that reduces false positives from the current overly-broad `\b(create|write|...)\b` pattern while maintaining safety:

```python
def _has_write_intent(prompt: str) -> bool:
    """
    Enhanced write operation detection with contextual understanding.
    The current implementation uses r"\\b(create|write|edit|modify|...)\\b" which
    matches write-related words ANYWHERE in the prompt, causing false positives
    on read-only queries like "show me how the code creates objects" or
    "explain what the write method does".

    This replacement:
    1. Checks for explicit imperative write commands (verb + code target noun)
    2. Excludes queries in known read-only phrasing patterns (question/explanation)
    3. Returns False for ambiguous cases where write words appear but no imperative
       command structure is detected (instead of the current blanket True)
    """
    lowered_prompt = prompt.lower().strip()

    # ── Layer 1: Explicit imperative write command patterns ──
    # These match when a write verb is used as an imperative/instruction
    # directly targeting a code entity (verb followed by a code noun object).
    # This eliminates false positives where "write"/"create" appear as nouns
    # or in descriptive/analytical contexts.
    explicit_write_patterns = [
        # Imperative: "create a file", "modify the code", "fix the following test"
        r"\b(create|write|edit|modify|update|delete|remove|refactor|rename|move|implement|fix|patch)\s+(the\s+)?(following\s+)?(file|files|code|test|tests|function|class|method|doc|docs|documentation|config|configuration|module|package|script|component)\b",
        # "add/insert a function/class/test" — add requires a direct object
        r"\b(add|insert)\s+(the\s+)?(following\s+)?(code|function|class|test|tests|documentation|module|package|file|files)\b",
        # "generate and save", "produce to file" — explicit save intent
        r"\b(generate|produce)\s+(and\s+)?(save|persist|store|write|output\s+to)\b",
        # "save/store/persist the result/output/code" — explicit persistence
        r"\b(save|store|persist)\s+(the\s+)?(result|output|code|file|changes|modification)\b",
        # "replace X with Y", "overwrite the file" — destructive operations
        r"\b(replace|overwrite)\s+.*\b(with|by)\b",
        # "remove/delete the file/function" (without a question context)
        r"\b(remove|delete)\s+(the\s+)?(file|files|directory|folder|code|function|class|method|module)\b",
    ]

    for pattern in explicit_write_patterns:
        if re.search(pattern, lowered_prompt, re.IGNORECASE):
            return True

    # ── Layer 2: Read-only context detection ──
    # If the prompt is phrased as a question, explanation request, or
    # analytical query, any write-related words are being used descriptively,
    # not as instructions to modify code.
    read_only_context_patterns = [
        # Questions about how to do something (learning, not doing)
        r"how\s+(to|do|can\s+i|does|should\s+i)\s+(write|create|modify|update|delete|remove|edit|implement|fix)",
        # Requests for examples or demonstrations
        r"(example|demonstration|sample|illustration)\s+(of|for|showing)\s+(writing|creating|modifying|updating|deleting|removing|how\s+to)",
        # Best practices / guidelines (knowledge, not action)
        r"(best\s+practice|guideline|recommendation|pattern|convention)\s+(for|about|on)\s+(writing|creating|modifying|updating|deleting|removing)",
        # Explanation requests
        r"(explain|describe|what\s+(does|is|are)|tell\s+me\s+about|show\s+me\s+how)\s+.*(write|create|modify|update|delete|remove|wrote|created|writes|creates)",
        # Documentation/reference queries
        r"(documentation|docs|reference|api)\s+(for|about|on)\s+(write|create|modify|update|delete|remove)",
        # Analytical/review queries: "analyze how X creates Y", "review the update logic"
        r"(analyze|review|examine|investigate|compare|find|search|list|count|check|verify|understand)\s+.*(write|create|modify|update|delete|remove)",
        # Past-tense or third-person: "where the code writes to disk", "how the factory creates objects"
        r"(where|how|when|why)\s+.*(writes|creates|modifies|updates|deletes|removes|wrote|created|modified|updated|deleted|removed)",
        # "the write method", "the create function" — referring to named entities
        r"(the\s+)?(write|create|modify|update|delete|remove)\s+(method|function|handler|callback|operation|routine|procedure|class|module|interface|trait|decorator)",
    ]

    # If ANY read-only context pattern matches, treat the entire prompt as read-only
    # regardless of whether individual write words appear. This is the key fix:
    # queries like "explain what the write method does" or "show me how the code
    # creates objects" will match read-only patterns and return False.
    for pattern in read_only_context_patterns:
        if re.search(pattern, lowered_prompt, re.IGNORECASE):
            return False

    # ── Layer 3: Ambiguous case handling ──
    # If write-related words appear but no explicit imperative command matched
    # (Layer 1) and no read-only context matched (Layer 2), we have an
    # ambiguous case. The current code returns True for ALL such cases
    # (any word in lowered_prompt from write_words → True), which is overly
    # conservative. Instead, we only return True if the write word appears
    # in a syntactic position suggesting an instruction (followed by a direct
    # object within 40 chars, similar to the current second pattern but broader).
    ambiguous_write_patterns = [
        # "write something", "create something" with a nearby target
        r"\b(create|write|edit|modify|update|delete|remove|refactor|rename|move|implement)\b.{0,40}\b(file|files|code|test|tests|function|class|method|doc|docs|documentation|config|module|package|component|script)\b",
        # Standalone imperative without explicit target but with instruction cues
        r"\b(please|kindly|make\s+sure|ensure)\s+.*\b(create|write|edit|modify|update|delete|remove)\b",
    ]

    for pattern in ambiguous_write_patterns:
        if re.search(pattern, lowered_prompt, re.IGNORECASE):
            return True

    # ── Layer 4: Default safe ──
    # If nothing matched, default to False (read-only). This is a deliberate
    # change from the current behavior which defaults to True when write words
    # appear. The rationale: the explicit and ambiguous patterns above already
    # catch genuine write intents; remaining cases are likely read-only queries
    # that happen to contain write-related words in passing. The LLM eligibility
    # classifier provides a second safety net for truly ambiguous edge cases.
    return False
```

> **Design Note**: The key behavioral change is Layer 2 (read-only context detection) and Layer 4 (default to False). The current code returns `True` for any prompt containing `\b(create|write|...)\b` regardless of context, which makes queries like "explain what the write method does" falsely classified as write operations. The new design reverses this: explicit imperative commands (Layer 1) return `True`, known read-only phrasing (Layer 2) returns `False`, and ambiguous cases default to `False` (Layer 4) with a secondary ambiguous-pattern check (Layer 3) to catch write intents that aren't imperative but still target code entities. The LLM eligibility classifier in `ConcurrencyEligibilityClassifier.is_eligible()` provides an additional safety net.

### 2. Improved Task Splitting with File Type Hints
**File**: `codebase_rag/orchestrator/task_splitter.py`

> **Design Note**: The current `_collect_scoped_files()` already falls back to `return all_files` when no scope paths are found. The proposed "Strategy 3: Default to all code files" is therefore redundant with existing behavior. The real improvement is Strategy 2 (file type hint filtering), which narrows the scope when the prompt mentions specific languages or file categories. **We do NOT propose truncating to `all_files[:500]`** — that would silently drop parts of the codebase and produce incomplete results. Instead, when all files are returned, the existing `CGR_PARALLEL_MAX_QUEUE_SIZE` check in `_run_interactive_loop` (which compares `preview_count` against `settings.CGR_PARALLEL_MAX_QUEUE_SIZE`) already prevents excessive parallelization by falling back to sequential execution.

Enhance `_collect_scoped_files()` with a file-type-hint intermediate strategy between scope-path matching and the full fallback:

```python
def _collect_scoped_files(self, prompt: str) -> list[Path]:
    """Enhanced file collection with file type hint filtering.

    Fallback order:
    1. Explicit scope paths from prompt (existing, unchanged)
    2. File type hints inferred from prompt language/category keywords (NEW)
    3. All code files (existing fallback — no truncation)
    """
    # Strategy 1: Extract explicit paths from prompt (unchanged from current)
    scope_paths = self._extract_scope_paths(prompt)

    if scope_paths:
        all_files = get_all_code_files(self.repo_path)
        scoped_files = [
            file_path for file_path in all_files
            if any(self._path_matches_scope(file_path, scope_path) for scope_path in scope_paths)
        ]
        if scoped_files:
            scoped_files.sort(key=lambda path: os.path.relpath(path, self.repo_path))
            logger.info(f"Scoped file split selected {len(scoped_files)} files from explicit paths")
            return scoped_files

    # Strategy 2: Analyze prompt for file type hints (NEW)
    # This reduces subtask count by narrowing to relevant file types
    # instead of returning all files, which may be thousands.
    extension_hints, name_pattern_hints = self._extract_file_type_hints(prompt)
    if extension_hints or name_pattern_hints:
        all_files = get_all_code_files(self.repo_path)
        hinted_files = _filter_files_by_hints(all_files, extension_hints, name_pattern_hints)
        if hinted_files:
            hinted_files.sort(key=lambda path: os.path.relpath(path, self.repo_path))
            logger.info(
                f"File type hints yielded {len(hinted_files)} files "
                f"(extensions: {extension_hints}, patterns: {name_pattern_hints})"
            )
            return hinted_files

    # Strategy 3: Fallback to all code files (unchanged from current)
    # NOTE: Do NOT truncate to all_files[:500] — that silently drops parts of
    # the codebase. The existing CGR_PARALLEL_MAX_QUEUE_SIZE check in
    # _run_interactive_loop handles excessive subtask counts correctly
    # by falling back to sequential execution with a clear log message.
    all_files = get_all_code_files(self.repo_path)
    logger.info(f"Using all {len(all_files)} code files (no scope paths or type hints found)")
    return all_files


def _filter_files_by_hints(
    all_files: list[Path],
    extension_hints: list[str],
    name_pattern_hints: list[str],
) -> list[Path]:
    """Filter files by extension hints (suffix-based) and name pattern hints (filename-based).

    This function correctly handles the two distinct types of hints:
    - Extension hints (e.g., '.py', '.js') are matched against f.suffix.lower()
    - Name pattern hints (e.g., '_test', 'test_', 'readme', 'config') are matched
      against f.name.lower() — NOT f.suffix, since f.suffix only contains the
      extension (e.g., '.py') not the full filename.
    """
    if not extension_hints and not name_pattern_hints:
        return all_files

    filtered = []
    for f in all_files:
        suffix_lower = f.suffix.lower()
        name_lower = f.name.lower()

        # Check extension hints against suffix
        ext_match = any(hint == suffix_lower for hint in extension_hints)

        # Check name pattern hints against filename
        name_match = any(hint in name_lower for hint in name_pattern_hints)

        if ext_match or name_match:
            filtered.append(f)

    return filtered


def _extract_file_type_hints(self, prompt: str) -> tuple[list[str], list[str]]:
    """Extract file type hints from prompt, returning extension hints and name pattern hints separately.

    Returns:
        Tuple of (extension_hints, name_pattern_hints) where:
        - extension_hints: pure file extensions like '.py', '.js' — matched via f.suffix
        - name_pattern_hints: substrings of filenames like '_test', 'test_', 'readme',
          'config' — matched via f.name (NOT f.suffix, which only contains the extension)

    This separation is critical: the previous version mixed these two types into a
    single list and checked all hints against f.suffix.lower(), causing filename
    patterns like '_test.py' and 'readme' to never match (since f.suffix is just '.py'
    or '.md', not the full filename).
    """
    lowered = prompt.lower()
    extension_hints: list[str] = []
    name_pattern_hints: list[str] = []

    # Language-specific extension hints (matched against f.suffix)
    if any(word in lowered for word in ['python', '.py', 'django', 'flask']):
        extension_hints.append('.py')
    if any(word in lowered for word in ['javascript', '.js', 'react', 'node', 'express']):
        extension_hints.extend(['.js', '.jsx', '.ts', '.tsx'])
    if any(word in lowered for word in ['java', '.java', 'spring', 'android']):
        extension_hints.append('.java')
    if any(word in lowered for word in ['c++', '.cpp', 'stl']):
        extension_hints.extend(['.cpp', '.h', '.hpp'])
    if any(word in lowered for word in ['go', '.go', 'golang']):
        extension_hints.append('.go')
    if any(word in lowered for word in ['rust', '.rs', 'cargo']):
        extension_hints.append('.rs')
    if any(word in lowered for word in ['c#', '.cs', 'csharp', '.net', 'asp.net']):
        extension_hints.append('.cs')

    # General name pattern hints (matched against f.name, NOT f.suffix)
    if 'test' in lowered or 'spec' in lowered:
        name_pattern_hints.extend(['_test', '_spec', 'test_', 'spec_', '.test', '.spec'])
    if 'config' in lowered or 'setting' in lowered:
        # Config files can be extension-based (.json, .yaml, .toml) or name-based (config, settings)
        extension_hints.extend(['.json', '.yaml', '.yml', '.toml', '.ini'])
        name_pattern_hints.extend(['config', 'settings', 'configuration'])
    if 'readme' in lowered:
        name_pattern_hints.extend(['readme'])
        extension_hints.extend(['.md', '.rst'])
    if 'doc' in lowered and 'documentation' not in lowered:
        # Avoid false positive on "documentation for" queries
        name_pattern_hints.extend(['doc', 'docs'])

    return extension_hints, name_pattern_hints
```

### 3. Enhanced LLM Eligibility Classification
**File**: `codebase_rag/orchestrator/concurrency_eligibility_classifier.py`

> **Critical Bug Fix**: The previous version of this spec contained `{prompt}` inside the `LLM_ELIGIBILITY_PROMPT` system prompt template. This is incorrect: `pydantic_ai.Agent` passes the system prompt and user prompt as **separate messages**. When `self.agent.run(prompt)` is called, the user prompt is sent as a `UserPromptPart` — it does NOT replace `{prompt}` in the system prompt template. The literal string `{prompt}` would appear in the system message seen by the LLM, creating a confusing double-prompt situation. The fix: **remove `{prompt}` from the system prompt entirely** — the user prompt is automatically provided by `pydantic_ai` when `agent.run(prompt)` is called.

Improve the LLM prompt by adding codebase context to the **system prompt only** (the user prompt is handled by `pydantic_ai` separately):

```python
# Enhanced LLM system prompt with codebase context
# NOTE: {prompt} is NOT included here. pydantic_ai sends the user prompt
# as a separate message via agent.run(prompt). Including {prompt} in the
# system prompt would create a confusing double-prompt where the LLM sees
# both "{prompt}" literally in the system message AND the actual user message.
LLM_ELIGIBILITY_PROMPT = """
You are a parallel task eligibility classifier for a codebase analysis system.
Evaluate if the user request (provided separately) can be split into independent,
non-overlapping subtasks that can be executed in parallel to speed up results.

Codebase Context:
- Total files: {file_count}
- Primary languages: {languages}
- Repository size: {repo_size}

Respond ONLY with a valid JSON object with three keys:
1. "eligible": boolean (true if request can be safely parallelized, false otherwise)
2. "confidence": float between 0.0 and 1.0 indicating confidence in this assessment
3. "reasoning": string explaining the decision briefly

Safety Rules:
- NEVER parallelize tasks that modify, create, delete, or update files/code
- ALWAYS parallelize read-only tasks that analyze multiple files or entities
- When uncertain, prefer sequential execution (eligible=false)
"""
```

Update the `_get_llm_eligibility` method to include codebase context in the **system prompt only**:

```python
async def _get_llm_eligibility(self, prompt: str) -> tuple[float, str]:
    """Run enhanced LLM analysis with codebase context."""
    if not self.agent:
        config = settings.active_orchestrator_config
        provider = get_provider_from_config(config)
        llm = provider.create_model(config.model_id)

        # Get codebase context (only if enabled and not already cached)
        if settings.CGR_PARALLEL_CODEBASE_CONTEXT:
            try:
                from codebase_rag.utils.path_utils import get_all_code_files
                repo_path = Path(settings.TARGET_REPO_PATH)
                all_files = get_all_code_files(repo_path)
                file_count = len(all_files)

                # Detect primary languages from file extensions
                extensions = [f.suffix.lower() for f in all_files if f.suffix]
                lang_counts: dict[str, int] = {}
                for ext in extensions:
                    lang = {
                        '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                        '.java': 'Java', '.cpp': 'C++', '.h': 'C++', '.go': 'Go',
                        '.rs': 'Rust', '.cs': 'C#', '.rb': 'Ruby', '.php': 'PHP'
                    }.get(ext, 'Other')
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

                primary_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                languages = ', '.join([lang for lang, count in primary_langs if count > 5])

                # Get repo size
                total_size = sum(f.stat().st_size for f in all_files if f.exists())
                repo_size = f"{total_size / (1024*1024):.1f}MB" if total_size > 0 else "unknown"

            except Exception as e:
                logger.warning(f"Failed to get codebase context: {e}")
                file_count = 0
                languages = "unknown"
                repo_size = "unknown"
        else:
            file_count = 0
            languages = "unknown"
            repo_size = "unknown"

        # Format system prompt with codebase context only (NO {prompt} placeholder)
        # The user prompt is sent separately by pydantic_ai via agent.run(prompt)
        system_prompt = self.LLM_ELIGIBILITY_PROMPT.format(
            file_count=file_count,
            languages=languages,
            repo_size=repo_size,
        )

        self.agent = Agent(
            model=llm,
            system_prompt=system_prompt,
            output_type=dict,
            retries=settings.AGENT_RETRIES,
        )

    try:
        result = await self.agent.run(prompt)
        result_data_raw = result.output
        if not isinstance(result_data_raw, dict):
            return 0.0, "llm_invalid_output"

        result_data = cast(dict[str, object], result_data_raw)
        confidence = max(
            0.0,
            min(1.0, self._coerce_float(result_data.get("confidence", 0.0))),
        )
        task_type = (
            result_data.get("task_type", "llm_analyzed")
            if result_data.get("eligible", False)
            else "llm_rejected"
        )
        # Log reasoning for debugging/auditability
        reasoning = result_data.get("reasoning", "")
        if reasoning:
            logger.debug(f"LLM eligibility reasoning: {reasoning}")

        return confidence, str(task_type)
    except Exception as e:
        logger.warning(
            f"LLM eligibility check failed: {str(e)}, falling back to sequential execution"
        )
        return 0.0, "llm_check_failed"
```

> **Latency Consideration**: Computing codebase context (iterating all files, computing file sizes via `f.stat().st_size`) adds latency to every eligibility check. On large repos (1000+ files), this could take 50-100ms. This is acceptable because:
> 1. The context is computed only once per agent initialization (cached in `self.agent`)
> 2. The benefit of better LLM classification outweighs the one-time overhead
> 3. The `CGR_PARALLEL_CODEBASE_CONTEXT` setting allows disabling this feature if latency is a concern
>
> However, `f.stat().st_size` and `f.exists()` calls for every file are expensive on network filesystems. Consider caching these values after the first computation.

### 4. User Override Mechanism
**File**: `codebase_rag/main.py`

Add explicit user override support in the parallel execution decision logic.

> **Critical Bug Fix**: `_normalize_parallel_config()` currently creates a new `ParallelExecutionConfig` by explicitly passing only the existing 6 fields. If `force_parallel` is added to the dataclass but not passed through `_normalize_parallel_config`, it will be **silently dropped** during normalization (defaulting to `False` regardless of the user's CLI input). The spec must include the `_normalize_parallel_config` update.

> **Safety Guard**: `force_parallel` must NOT override write-operation safety. Even with `--force-parallel`, if `_has_write_intent()` detects a write operation, parallel execution must be blocked to prevent data corruption. `force_parallel` only bypasses the LLM eligibility classifier's confidence threshold — not the write-safety gate.

```python
# 1. Add force_parallel to ParallelExecutionConfig
@dataclass(frozen=True)
class ParallelExecutionConfig:
    worker_count: int | None = None
    auto_split: bool = settings.CGR_AUTO_SPLIT_ENABLED
    no_parallel: bool = False
    dry_run: bool = False
    scheduling_strategy: str = "fifo"
    doc_workspace: str = "default"
    force_parallel: bool = False  # NEW: Explicit user override

# 2. CRITICAL: Update _normalize_parallel_config to pass force_parallel through
# Without this, force_parallel is silently dropped during normalization,
# making the feature completely broken.
def _normalize_parallel_config(
    parallel_config: ParallelExecutionConfig | None,
) -> ParallelExecutionConfig:
    normalized = parallel_config or ParallelExecutionConfig()
    scheduling_strategy = normalized.scheduling_strategy.lower()
    if scheduling_strategy not in {"fifo", "round-robin"}:
        raise ValueError("Invalid scheduling strategy. Use 'fifo' or 'round-robin'.")
    return ParallelExecutionConfig(
        worker_count=normalized.worker_count,
        auto_split=normalized.auto_split,
        no_parallel=normalized.no_parallel,
        dry_run=normalized.dry_run,
        scheduling_strategy=scheduling_strategy,
        doc_workspace=normalized.doc_workspace,
        force_parallel=normalized.force_parallel,  # NEW: Must be passed through
    )

# 3. Update the parallel execution decision logic in _run_interactive_loop
# force_parallel bypasses LLM eligibility but NOT write-safety checks
if normalized_parallel_config.no_parallel:
    logger.info("Parallel execution skipped due to explicit sequential override")
elif has_write_operations:
    # SAFETY: Write operations ALWAYS block parallel execution.
    # force_parallel does NOT override this — it only bypasses the
    # LLM eligibility threshold, not the write-safety gate.
    # Attempting to parallelize write operations risks data corruption
    # from concurrent file modifications.
    logger.info(
        "Parallel execution skipped: contains write operations "
        "(cannot be overridden by --force-parallel for safety)"
    )
elif normalized_parallel_config.force_parallel:
    # force_parallel bypasses the LLM eligibility classifier's threshold.
    # It should only be used when the user knows their task is read-only
    # but the classifier incorrectly rejects it (e.g., low confidence).
    logger.warning(
        "Parallel execution forced by user override (--force-parallel). "
        "Write safety checks are still enforced; this only bypasses "
        "the LLM eligibility threshold."
    )
    eligible, task_type, confidence = True, "user_forced", 1.0
# ... rest of existing eligibility logic
```

Update CLI to support force parallel flag:

```python
# Add to CLI options in codebase_rag/cli.py
force_parallel: bool = typer.Option(
    False,
    "--force-parallel",
    help="Force parallel execution bypassing LLM eligibility threshold "
         "(write-safety checks are still enforced)",
),

# Pass to ParallelExecutionConfig (in the start command)
parallel_config = ParallelExecutionConfig(
    worker_count=parallel_workers,
    auto_split=auto_split,
    no_parallel=no_parallel,
    dry_run=parallel_dry_run,
    scheduling_strategy=normalized_scheduling_strategy,
    doc_workspace=doc_workspace,
    force_parallel=force_parallel,  # NEW
)
```

### 5. Dynamic Threshold Adjustment
**File**: `codebase_rag/orchestrator/concurrency_eligibility_classifier.py`

> **Critical Design Issues in Previous Version**:
> 1. **Duplicate calibration**: The existing `ConcurrencyEligibilityClassifier.is_eligible()` already implements dynamic calibration via confidence multipliers (step 6: `confidence *= 1.1` when success rate > 0.9, `confidence *= 0.9` when success rate < 0.6). The previous spec proposed a separate `_adjust_threshold_based_on_success()` method that adjusts the **threshold** instead of the **confidence**. Having BOTH creates a double-effect interaction: high success rate → confidence boosted by 1.1 AND threshold lowered to 0.56 → overly aggressive parallelization. We must **replace** the existing calibration, not add a duplicate.
> 2. **Naming collision**: The previous spec introduced `self.base_threshold` and `self.current_threshold`, but the existing code uses `self.threshold` (not `base_threshold` or `current_threshold`). Introducing new names creates confusion. We reuse `self.threshold` as the base value and introduce `self._effective_threshold` for the dynamically adjusted value.
> 3. **Minimum data points**: The previous spec used 5 data points minimum, but the existing code uses 10. We align to 10 for consistency with the existing proven threshold.

Replace the existing confidence-multiplier calibration (step 6 in `is_eligible()`) with threshold-based adjustment. This is a cleaner approach: adjusting the bar (threshold) is more intuitive and observable than adjusting the score (confidence), and produces clearer log messages.

```python
def __init__(self):
    self.enabled: bool = getattr(settings, "CGR_AUTO_PARALLEL_ENABLED", True)
    # self.threshold is the BASE threshold (unchanged from current code).
    # It is set from CGR_PARALLEL_ELIGIBILITY_THRESHOLD and stays fixed.
    self.threshold: float = getattr(
        settings, "CGR_PARALLEL_ELIGIBILITY_THRESHOLD", 0.7
    )
    # self._effective_threshold is the dynamically adjusted threshold that
    # actually gets used in the eligibility decision. It starts equal to
    # self.threshold and may be adjusted based on historical success rates.
    # This replaces the previous approach of adjusting confidence scores
    # via multipliers, which produced opaque "effective threshold" changes.
    self._effective_threshold: float = self.threshold
    self.min_subtask_count: int = 2
    self.agent: Agent | None = None
    # Existing success_rate_tracker (unchanged) — keeps last 100 results per type
    self.success_rate_tracker: dict[str, list[bool]] = {}
    # Adaptive adjustment enabled via CGR_PARALLEL_ADAPTIVE_THRESHOLD config
    self.adaptive_adjustment_enabled: bool = getattr(
        settings, "CGR_PARALLEL_ADAPTIVE_THRESHOLD", True
    )

def _adjust_threshold_based_on_success(self, task_type: str) -> float:
    """Dynamically adjust the effective eligibility threshold based on historical success rates.

    This REPLACES the existing confidence-multiplier calibration in is_eligible()
    (which adjusted confidence by *1.1 or *0.9). Threshold-based adjustment is:
    - More intuitive: lowering the bar vs. boosting the score
    - More observable: log messages show explicit threshold changes
    - Less prone to double-effect: only one mechanism, not two

    The effective threshold is adjusted relative to self.threshold (the base):
    - High success rate (>0.85): lower threshold to enable more parallelization
    - Good success rate (>0.7): keep base threshold
    - Moderate success rate (>0.5): slightly raise threshold
    - Low success rate (<0.5): significantly raise threshold

    Args:
        task_type: The task type to adjust threshold for

    Returns:
        The effective threshold to use for this eligibility decision
    """
    if not self.adaptive_adjustment_enabled:
        return self.threshold

    if task_type not in self.success_rate_tracker:
        return self._effective_threshold

    successes = self.success_rate_tracker[task_type]
    # Aligned with existing code's minimum of 10 data points (not 5).
    # The previous spec used 5, but the existing code requires 10 for
    # more reliable calibration. Using fewer data points produces noisy
    # threshold adjustments that may over-correct on limited evidence.
    if len(successes) < 10:
        return self._effective_threshold

    success_rate = sum(successes) / len(successes)

    # Adjust effective threshold based on success rate
    if success_rate >= 0.85:
        # High success rate: lower threshold to enable more parallelization
        new_threshold = max(0.5, self.threshold * 0.8)
    elif success_rate >= 0.7:
        # Good success rate: keep base threshold
        new_threshold = self.threshold
    elif success_rate >= 0.5:
        # Moderate success rate: slightly raise threshold
        new_threshold = min(0.8, self.threshold * 1.1)
    else:
        # Low success rate: significantly raise threshold
        new_threshold = min(0.9, self.threshold * 1.3)

    if new_threshold != self._effective_threshold:
        logger.info(
            f"Adaptive threshold adjusted for {task_type}: "
            f"{self._effective_threshold:.2f} → {new_threshold:.2f} "
            f"(success rate: {success_rate:.2%}, base threshold: {self.threshold:.2f})"
        )
        self._effective_threshold = new_threshold

    return self._effective_threshold
```

**Update `is_eligible()` to use threshold-based adjustment**: Replace step 6 (the existing confidence-multiplier calibration) with a call to `_adjust_threshold_based_on_success()`, and use the effective threshold in the final eligibility decision instead of `self.threshold`:

```python
# In is_eligible(), REPLACE step 6 (existing confidence multiplier calibration):
#
#   OLD (to be removed):
#     if (task_type in self.success_rate_tracker
#         and len(self.success_rate_tracker[task_type]) >= 10):
#         success_rate = sum(self.success_rate_tracker[task_type]) / len(...)
#         if success_rate > 0.9:
#             confidence = min(1.0, confidence * 1.1)
#         elif success_rate < 0.6:
#             confidence = max(0.0, confidence * 0.9)
#
#   NEW (to be added):
#     effective_threshold = self._adjust_threshold_based_on_success(task_type)
#
# And in the final eligibility decision, REPLACE:
#     if confidence >= self.threshold:
#   WITH:
#     if confidence >= effective_threshold:
#
# This eliminates the double-effect problem where both confidence AND threshold
# were being adjusted, producing overly aggressive or overly conservative results.
```

## Configuration Updates

> **Critical Issue**: `AppConfig` uses `pydantic_settings.BaseSettings`, which **silently ignores** any environment variable not declared as a class field. The previous version only listed new env vars for `.env.example` but did not declare them in `AppConfig`. Without `AppConfig` field declarations, these settings will never be read by the application — they'll be silently ignored regardless of what's in `.env`. All new settings must be added to `codebase_rag/config.py`'s `AppConfig` class.

### New Environment Variables

**Step 1**: Add field declarations to `AppConfig` in `codebase_rag/config.py` (REQUIRED — without this, env vars are silently ignored):

```python
# Add these fields to the AppConfig class in codebase_rag/config.py

# ─────────────────────────────────────────────────────────
# Enhanced Parallel Execution Configuration (NEW)
# ─────────────────────────────────────────────────────────

# Strict write detection mode: when True, uses the current broad regex patterns
# (more false positives, more conservative). When False (default), uses the
# enhanced Layer 1-4 detection from Section 1 of this spec.
CGR_PARALLEL_WRITE_DETECTION_STRICT: bool = False

# Minimum subtasks required to justify parallel execution overhead
CGR_PARALLEL_MIN_SUBTASKS: int = Field(default=2, gt=0)

# Enable adaptive threshold adjustment based on historical parallel execution success rates
# When True, replaces the existing confidence-multiplier calibration with threshold-based adjustment
CGR_PARALLEL_ADAPTIVE_THRESHOLD: bool = True

# Enable file type hint extraction for better task splitting
# When True, _extract_file_type_hints() is used in _collect_scoped_files() Strategy 2
CGR_PARALLEL_FILE_TYPE_HINTS: bool = True

# Include codebase context (file count, languages, repo size) in LLM eligibility analysis
# When True, codebase stats are computed once per agent initialization and included in the system prompt
CGR_PARALLEL_CODEBASE_CONTEXT: bool = True
```

**Step 2**: Add to `.env.example` (for documentation/discovery):

```env
# ─────────────────────────────────────────────────────────
# Enhanced Parallel Execution Configuration
# ─────────────────────────────────────────────────────────

# Strict write detection mode (default: false)
# When true, uses current broad regex (more false positives, more conservative)
# When false, uses enhanced Layer 1-4 detection with contextual understanding
CGR_PARALLEL_WRITE_DETECTION_STRICT=false

# Minimum subtasks required for parallelization (default: 2, min: 1)
CGR_PARALLEL_MIN_SUBTASKS=2

# Enable adaptive threshold adjustment (default: true)
# Replaces confidence-multiplier calibration with threshold-based adjustment
CGR_PARALLEL_ADAPTIVE_THRESHOLD=true

# Enable file type hint extraction for task splitting (default: true)
# Adds Strategy 2 (language/category filtering) to _collect_scoped_files()
CGR_PARALLEL_FILE_TYPE_HINTS=true

# Include codebase context in LLM eligibility analysis (default: true)
# Adds file count, languages, repo size to LLM system prompt
CGR_PARALLEL_CODEBASE_CONTEXT=true

# Force parallel execution override (CLI flag only, no env var)
# Use --force-parallel on the command line; not configurable via environment
```

### Updated Defaults (REQUIRE CODE CHANGES)

> **Note**: These are not just `.env.example` documentation changes — they require updating the default values in `AppConfig` class fields in `codebase_rag/config.py`. Simply changing `.env.example` without updating `AppConfig` defaults will have no effect on new installations that don't have a `.env` file.

- **Lower default threshold from 0.7 to 0.6**: Change `CGR_PARALLEL_ELIGIBILITY_THRESHOLD: float = 0.7` → `CGR_PARALLEL_ELIGIBILITY_THRESHOLD: float = 0.6` in `AppConfig`
- **Increase maximum queue size from 100 to 200**: Change `CGR_PARALLEL_MAX_QUEUE_SIZE: int = 100` → `CGR_PARALLEL_MAX_QUEUE_SIZE: int = 200` in `AppConfig`
- **Enable adaptive threshold adjustment by default**: Already covered by `CGR_PARALLEL_ADAPTIVE_THRESHOLD: bool = True` (new field above)

## Testing Strategy
### Unit Tests
1. **Write Detection Tests**: Verify enhanced detection correctly identifies true write operations while allowing safe read-only queries containing write-related words
2. **Task Splitting Tests**: Validate fallback strategies generate sufficient subtasks for various prompt types
3. **LLM Classification Tests**: Ensure codebase context improves confidence scores for eligible tasks
4. **User Override Tests**: Confirm force-parallel flag bypasses all eligibility checks

### Integration Tests
1. **End-to-End Parallel Execution**: Test complete workflow with various query types
2. **Performance Benchmarking**: Measure parallelization rate improvement and execution speedup
3. **Safety Validation**: Ensure no write operations are accidentally parallelized

### Edge Case Tests
1. **Empty Repository**: Handle repositories with no code files gracefully
2. **Very Large Repositories**: Test with repositories containing 1000+ files
3. **Ambiguous Queries**: Verify conservative behavior for unclear user intents

## Migration Plan
### Backward Compatibility
- All existing configuration options remain functional
- Default behavior becomes more permissive but maintains safety guarantees
- No breaking changes to existing APIs or user workflows

### Rollout Strategy
1. **Phase 1**: Deploy enhanced write detection and task splitting (immediate impact)
2. **Phase 2**: Enable adaptive threshold adjustment with monitoring
3. **Phase 3**: Roll out LLM enhancements with codebase context
4. **Phase 4**: Add user override mechanisms and final optimizations

## Performance Impact
### Expected Improvements
- **Parallelization Rate**: Increase from ~30% to ~70% of eligible queries
- **Execution Speed**: Maintain 4-8x speedup for parallelized tasks
- **False Positive Reduction**: Reduce write operation false positives by 80%

### Resource Usage
- **Memory**: Minimal increase due to codebase context caching
- **CPU**: Slight increase during eligibility classification (offset by parallel execution benefits)
- **API Calls**: Additional LLM calls for eligibility classification (justified by performance gains)

## Success Metrics
1. **Parallelization Rate**: Percentage of queries that trigger parallel execution
2. **Execution Time Reduction**: Average speedup factor for parallelized queries  
3. **User Satisfaction**: Reduction in user complaints about slow execution
4. **Safety Incidents**: Zero write operation parallelization incidents

## Future Enhancements
1. **Machine Learning Classifier**: Replace rule-based write detection with ML model
2. **Dynamic Worker Scaling**: Adjust worker count based on real-time system load
3. **Query Caching**: Cache eligibility decisions for frequently asked questions
4. **User Feedback Loop**: Allow users to provide feedback on parallelization decisions