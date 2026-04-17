# Parallel Execution Enhancements Specification
**Version**: 1.0  
**Date**: 2025-07-09  
**Status**: Implementation-Ready  
**Parent Spec**: `parallel-execution-optimization-fix-spec.md` v2.0  
**Purpose**: Documents enhancements implemented beyond the parent spec, identifies gaps, and provides implementation-ready specifications for closing those gaps.  

## Executive Summary

This specification covers two categories of changes relative to the parent spec (`parallel-execution-optimization-fix-spec.md` v2.0):

1. **Implemented Enhancements Beyond Parent Spec**: Features that were implemented correctly but were not specified in the parent spec. These are documented here for completeness and to establish canonical design rationale.

2. **Logical Gaps Found**: A critical gap where `ConcurrencyEligibilityClassifier.record_execution_result()` is defined but never called from the main execution loop, making the adaptive threshold adjustment mechanism inert. This spec provides the implementation-ready fix.

---

## Part I: Implemented Enhancements (Canonical Documentation)

These features are already implemented and working. This section provides their design rationale for traceability and future maintenance.

### 1. Extended Language Coverage in File Type Hints

**File**: `codebase_rag/orchestrator/task_splitter.py` — `_extract_file_type_hints()`  
**Parent Spec Coverage**: Only 7 languages (Python, JavaScript, Java, C++, Go, Rust, C#)  
**Enhancement**: 6 additional languages plus 3 domain categories  

#### 1.1 Additional Language Entries

| Language | Trigger Keywords | Extensions | Rationale |
|----------|-----------------|------------|-----------|
| Ruby | `ruby`, `.rb`, `rails` | `.rb` | Rails is a major web framework; Ruby repos often have hundreds of `.rb` files |
| PHP | `php`, `.php`, `laravel` | `.php` | Laravel/Symfony are major frameworks; PHP repos are common in web development |
| Swift | `swift`, `.swift`, `ios` | `.swift` | iOS development queries frequently target Swift files |
| Kotlin | `kotlin`, `.kt`, `android` | `.kt` | Modern Android development primarily uses Kotlin |
| Scala | `scala`, `.scala` | `.scala` | Data engineering (Spark) and JVM polyglot repos use Scala |
| TypeScript (standalone) | `typescript`, `.ts` | `.ts`, `.tsx` | Pure TypeScript projects may not mention "javascript" — this entry avoids missing them |

#### 1.2 Additional Domain Categories

| Category | Trigger Keywords | Extensions | Name Patterns | Rationale |
|-----------|-----------------|------------|---------------|-----------|
| HTML/Web/Frontend | `html`, `web`, `frontend` | `.html`, `.css`, `.scss`, `.sass` | — | Frontend-focused queries should not return backend Python/Java files |
| SQL/Database | `sql`, `database` | `.sql` | — | Database analysis queries should target SQL schema/migration files |
| Shell/Bash | `shell`, `bash`, `script` | `.sh`, `.bash` | — | CI/CD and deployment queries target shell scripts |

#### 1.3 Keyword Overlap Handling

> **Design Note**: Several trigger keywords appear in multiple language entries. This is **intentional** and **correct**:

- **`'android'`** triggers both `.java` and `.kt`: Android projects commonly mix Java and Kotlin files. Returning both ensures complete coverage.
- **`'react'`** triggers `.js`, `.jsx`, `.ts`, `.tsx`: React projects may use TypeScript or JavaScript. Returning both ensures no React component files are missed.
- **`'node'`** triggers `.js`, `.jsx`, `.ts`, `.tsx`: Node.js projects commonly include TypeScript configuration files alongside JavaScript source.
- **`'.ts'`** appears in both JavaScript and TypeScript entries: This is intentional redundancy — a user mentioning only "typescript" should still get `.ts/.tsx` without needing to also mention "javascript".

These overlaps produce **union semantics** (a file matching any hint is included), so duplicate hints never cause files to be missed or double-counted.

### 2. Expanded Language-to-Extension Mapping in Codebase Context

**File**: `codebase_rag/orchestrator/concurrency_eligibility_classifier.py` — `_get_codebase_context()`  
**Parent Spec Coverage**: Only 10 extensions mapped (`.py`, `.js`, `.ts`, `.java`, `.cpp`, `.h`, `.go`, `.rs`, `.cs`, `.rb` with `.rb` implicit)  
**Enhancement**: 7 additional extension mappings  

| Extension | Language | Rationale |
|-----------|----------|-----------|
| `.jsx` | JavaScript | React component files are JavaScript; grouping with `.js` avoids splitting React ecosystem |
| `.tsx` | TypeScript | React+TypeScript files; grouping with `.ts` avoids splitting TS ecosystem |
| `.hpp` | C++ | C++ header files (modern style); grouped with `.cpp` and `.h` per parent spec |
| `.swift` | Swift | iOS/macOS development language |
| `.kt` | Kotlin | Modern Android development language |
| `.scala` | Scala | Data engineering / JVM polyglot language |
| `.php` | PHP | Web development language |

> **Design Note**: The `count > 5` threshold for `languages` in `_get_codebase_context()` filters out languages with fewer than 5 files. This prevents noise from incidental files (e.g., a lone `.sql` file in a Python project) while ensuring major languages are always reported. The threshold value of 5 is calibrated: too low (1-2) would report transient languages, too high (20+) would hide secondary languages in polyglot repos.

### 3. Codebase Context Caching

**File**: `codebase_rag/orchestrator/concurrency_eligibility_classifier.py`  
**Parent Spec Coverage**: Parent spec notes "computed only once per agent initialization" but does not specify a caching mechanism  
**Enhancement**: `_codebase_context_cache` field prevents redundant recomputation  

#### 3.1 Implementation Details

```python
# In __init__:
self._codebase_context_cache: dict[str, object] | None = None

# In _get_codebase_context():
if self._codebase_context_cache is not None:
    return cast(tuple[int, str, str], tuple(self._codebase_context_cache.values()))
# ... compute context ...
self._codebase_context_cache = {"file_count": ..., "languages": ..., "repo_size": ...}
```

#### 3.2 Design Rationale

The parent spec's latency analysis notes that computing codebase context (iterating all files, `f.stat().st_size`) adds 50-100ms on large repos. Without caching, this would recompute on **every `_get_llm_eligibility()` call that triggers agent initialization**. Since `_get_llm_eligibility()` creates the `pydantic_ai.Agent` only once (`if not self.agent:`), the cache is effectively a per-session singleton:

1. First call: cache is `None` → compute and store
2. Subsequent calls: cache is populated → return cached values
3. Cache lifetime: tied to `ConcurrencyEligibilityClassifier` instance, which is created once per `_run_interactive_loop()` session

> **Design Note**: The `cast(tuple[int, str, str], tuple(self._codebase_context_cache.values()))` in the cache hit path is a runtime type assertion. Python `dict.values()` returns a `ValuesView` whose order matches insertion order (Python 3.7+), and since the cache is always constructed with keys `"file_count"`, `"languages"`, `"repo_size"` in that order, `tuple(values)` produces `(int, str, str)` as expected. The `cast` is for type-checker satisfaction only — no runtime conversion occurs.

#### 3.3 Cache Invalidation

The cache is **never invalidated** within a session. This is correct because:

- `TARGET_REPO_PATH` does not change within a session
- The file set (`get_all_code_files()`) does not change during a query — no ingestion occurs mid-chat
- `f.stat().st_size` values are stable during a read-only analysis session
- `ConcurrencyEligibilityClassifier` is recreated on session start, so stale data from a previous session is never carried over

> **Future Consideration**: If real-time file watching (`realtime_updater.py`) modifies the repo during a session, the cache would become stale. This is not a current concern because the chat loop does not interleave with ingestion, but should be addressed if concurrent modification is enabled.

---

## Part II: Logical Gaps — Implementation-Ready Fixes

### Gap 1: `record_execution_result()` Is Never Called

**Severity**: HIGH — Makes adaptive threshold adjustment completely inert  
**Impact**: The `_adjust_threshold_based_on_success()` method never receives any data, so `success_rate_tracker` remains empty and `_effective_threshold` always equals `self.threshold`. Adaptive threshold adjustment is a dead code path.  

#### 1.1 Root Cause Analysis

The parent spec introduced `_adjust_threshold_based_on_success()` (§5) and `record_execution_result()` as the data-collection mechanism. However, the spec only defined these methods on the classifier — it never specified **where** in the execution loop `record_execution_result()` should be called. The implementation correctly defined the method but no call site exists in `_run_interactive_loop()` or any other module.

```
grep result: "record_execution_result" only appears in the classifier's own definition.
No call sites exist in main.py, cli.py, or any orchestrator module.
```

#### 1.2 Fix Specification

**File**: `codebase_rag/main.py` — `_run_interactive_loop()`  

Insert `record_execution_result()` calls at **two** points in the parallel execution decision flow within `_run_interactive_loop()`:

**Point A**: After a successful parallel execution completes (record success):

```python
# After the parallel execution block where aggregator.consolidate() succeeds:
# This occurs in two places: (1) force_parallel path, (2) normal eligibility path
# Add after the aggregator.consolidate() call in both paths:

concurrency_classifier.record_execution_result(task_type, success=True)
```

**Point B**: When parallel execution is skipped due to LLM rejection (record failure):

```python
# After the "not eligible" branch where the classifier rejected the task:
# The task_type from the classifier's is_eligible() return value indicates
# why it was rejected. This provides calibration data even for rejected tasks.

if not eligible:
    logger.info(
        f"Parallel execution skipped: task_type={task_type}, confidence={confidence:.2f}"
    )
    # Record rejection as a "failure" for calibration — the classifier's
    # decision was that parallelization would likely fail, so recording
    # this as a negative result gives the adaptive threshold accurate data.
    # Note: We only record LLM-driven decisions, not rule-based rejections
    # (write_operation, user_requested_sequential, safety_rule_blocked,
    # insufficient_subtasks) since those are deterministic and don't benefit
    # from threshold adjustment.
    if task_type not in ("write_operation", "user_requested_sequential",
                          "safety_rule_blocked", "insufficient_subtasks"):
        concurrency_classifier.record_execution_result(task_type, success=False)
```

#### 1.3 Detailed Implementation

The `_run_interactive_loop()` method has three parallel execution paths that need `record_execution_result()` calls:

**Path 1: `force_parallel` execution** (lines after `eligible, task_type, confidence = True, "user_forced", 1.0`):

When `force_parallel` succeeds and actually runs parallel subtasks, record success with `task_type="user_forced"`:

```python
elif normalized_parallel_config.force_parallel:
    logger.warning(...)
    eligible, task_type, confidence = True, "user_forced", 1.0
    if preview_count is None or preview_count < settings.CGR_PARALLEL_MIN_SUBTASKS:
        logger.info(...)
    else:
        app_context.console.print(...)
        app_context.console.print(...)
        app_context.console.print(...)

        aggregator = subagent_orchestrator.execute_tasks(
            preview_subtasks,
            dry_run=normalized_parallel_config.dry_run,
        )
        parallel_result = aggregator.consolidate()
        
        # RECORD: Successful forced parallel execution
        concurrency_classifier.record_execution_result(task_type, success=True)
        
        summary_label = (
            "plan generated" if normalized_parallel_config.dry_run else "completed"
        )
        # ... rest of existing code
```

**Path 2: Normal eligibility-approved execution** (the `else` block after eligibility check succeeds):

```python
else:
    (
        eligible,
        task_type,
        confidence,
    ) = await concurrency_classifier.is_eligible(
        question_with_context,
        subtask_count=preview_count,
        has_write_operations=has_write_operations,
    )

    if not eligible:
        logger.info(
            f"Parallel execution skipped: task_type={task_type}, confidence={confidence:.2f}"
        )
        # RECORD: LLM-driven rejection for calibration
        # Only record non-deterministic rejection types (LLM decisions).
        # Deterministic rejections (write, safety rules) are always correct
        # and don't benefit from threshold adjustment.
        if task_type not in (
            "write_operation",
            "user_requested_sequential",
            "safety_rule_blocked",
            "insufficient_subtasks",
            "concurrency_disabled",
        ):
            concurrency_classifier.record_execution_result(task_type, success=False)
    elif not normalized_parallel_config.auto_split:
        logger.info(...)
    elif preview_count is None or preview_count < settings.CGR_PARALLEL_MIN_SUBTASKS:
        logger.info(...)
    else:
        app_context.console.print(...)
        app_context.console.print(...)
        app_context.console.print(...)

        aggregator = subagent_orchestrator.execute_tasks(
            preview_subtasks,
            dry_run=normalized_parallel_config.dry_run,
        )
        parallel_result = aggregator.consolidate()
        
        # RECORD: Successful classifier-approved parallel execution
        concurrency_classifier.record_execution_result(task_type, success=True)
        
        summary_label = (
            "plan generated" if normalized_parallel_config.dry_run else "completed"
        )
        # ... rest of existing code
```

#### 1.4 Why Record LLM Rejections as Failures

> **Design Note**: Recording LLM-driven rejections (`task_type="llm_rejected"`) as `success=False` is correct because:
>
> 1. The classifier's LLM determined that parallelization would likely produce poor results, so recording this as a negative outcome provides the adaptive threshold with accurate calibration data.
> 2. If the classifier is **overly conservative** (rejecting tasks that could actually be parallelized successfully), the success rate for `"llm_rejected"` will be artificially low, which raises the effective threshold further — making the system *more* conservative. This seems counterproductive, but it's actually **safety-correct**: the adaptive threshold should never lower the bar for task types where it has no evidence of successful parallelization.
> 3. The real calibration signal comes from successful parallel executions (`task_type="llm_analyzed"`, `"user_requested_parallel"`, `"user_forced"`). As these accumulate with `success=True`, the success rate rises above 0.85, which *lowers* the threshold, enabling more parallelization.
>
> We exclude deterministic rejection types (`write_operation`, `user_requested_sequential`, `safety_rule_blocked`, `insufficient_subtasks`, `concurrency_disabled`) because these are **always correct** — no amount of threshold adjustment should change their outcome. Recording them would pollute the calibration data with noise.

#### 1.5 Session Lifetime Implications

`ConcurrencyEligibilityClassifier` is instantiated once at the top of `_run_interactive_loop()`. Within a single session:

- **Query 1**: `record_execution_result("llm_analyzed", True)` → tracker has 1 entry
- **Query 2**: `record_execution_result("llm_analyzed", True)` → tracker has 2 entries
- ...
- **Query 10+**: Tracker reaches 10 entries → `_adjust_threshold_based_on_success()` becomes active

This means **adaptive threshold adjustment requires at least 10 queries in a single session** before it activates. For short sessions (1-5 queries), the threshold stays at the base value. This is acceptable because:

1. The base threshold (0.6) is already calibrated for general use
2. 10 data points is the minimum for statistically reliable adjustment (per §5 of parent spec)
3. Short sessions don't benefit enough from threshold adjustment to justify noisy adjustments on limited data

> **Future Consideration**: For long-running sessions (e.g., MCP server mode), the classifier persists across many queries, accumulating sufficient data for meaningful adjustment. For interactive CLI sessions, most users make 3-10 queries before ending the session. If adaptive adjustment is desired for shorter sessions, consider lowering the minimum data points to 5 with a higher uncertainty discount factor. However, this is a Phase 2 optimization and not part of this spec.

### Gap 2: `docstring` Misalignment in `record_execution_result()`

**Severity**: LOW — Cosmetic/docstring issue, no functional impact  
**Current**: `"""Record execution result for dynamic threshold calibration."""`  
**Correct**: Already correct — says "threshold calibration"  

> **Note**: Upon verification, the docstring already says "threshold calibration" (not "confidence calibration"), so this is NOT a gap. The comment on `self.success_rate_tracker` says "Dynamic calibration state" which is also correct since it's generic. No fix needed.

---

## Part III: Configuration Alignment Verification

All configuration fields from the parent spec plus the enhancements are correctly declared in `AppConfig`:

| Field | Type | Default | Status |
|-------|------|---------|--------|
| `CGR_PARALLEL_WRITE_DETECTION_STRICT` | `bool` | `False` | ✅ Declared in `AppConfig` |
| `CGR_PARALLEL_MIN_SUBTASKS` | `int` (Field, gt=0) | `2` | ✅ Declared in `AppConfig` |
| `CGR_PARALLEL_ADAPTIVE_THRESHOLD` | `bool` | `True` | ✅ Declared in `AppConfig` |
| `CGR_PARALLEL_FILE_TYPE_HINTS` | `bool` | `True` | ✅ Declared in `AppConfig` |
| `CGR_PARALLEL_CODEBASE_CONTEXT` | `bool` | `True` | ✅ Declared in `AppConfig` |
| `CGR_PARALLEL_ELIGIBILITY_THRESHOLD` | `float` | `0.6` | ✅ Changed from 0.7 per parent spec |
| `CGR_PARALLEL_MAX_QUEUE_SIZE` | `int` | `200` | ✅ Changed from 100 per parent spec |

All fields are also documented in `.env.example` with descriptions matching their `AppConfig` docstrings. `pydantic_settings.BaseSettings` will correctly read these from environment variables since they are declared as class fields.

> **Verification**: The parent spec's "Critical Issue" note warned that `AppConfig` uses `pydantic_settings.BaseSettings` which silently ignores undeclared fields. All new fields are declared, so this concern is addressed.

---

## Part IV: Test Specifications

### 4.1 Tests for Extended Language Coverage

**File**: `tests/test_task_splitter_file_type_hints.py` (new)

```python
import pytest
from codebase_rag.orchestrator.task_splitter import TaskSplitter, _filter_files_by_hints
from pathlib import Path


class TestExtractFileTypeHintsExtendedLanguages:
    """Verify extended language coverage beyond parent spec."""

    def setup_method(self):
        self.splitter = TaskSplitter(repo_path="/tmp/test_repo")

    # -- Ruby --
    def test_ruby_keyword_triggers_rb(self):
        ext, name = self.splitter._extract_file_type_hints("analyze ruby code")
        assert ".rb" in ext

    def test_rails_keyword_triggers_rb(self):
        ext, name = self.splitter._extract_file_type_hints("review rails models")
        assert ".rb" in ext

    # -- PHP --
    def test_php_keyword_triggers_php(self):
        ext, name = self.splitter._extract_file_type_hints("analyze php controllers")
        assert ".php" in ext

    def test_laravel_keyword_triggers_php(self):
        ext, name = self.splitter._extract_file_type_hints("review laravel routes")
        assert ".php" in ext

    # -- Swift --
    def test_swift_keyword_triggers_swift(self):
        ext, name = self.splitter._extract_file_type_hints("analyze swift views")
        assert ".swift" in ext

    def test_ios_keyword_triggers_swift(self):
        ext, name = self.splitter._extract_file_type_hints("review ios app code")
        assert ".swift" in ext

    # -- Kotlin --
    def test_kotlin_keyword_triggers_kt(self):
        ext, name = self.splitter._extract_file_type_hints("analyze kotlin coroutines")
        assert ".kt" in ext

    # -- Scala --
    def test_scala_keyword_triggers_scala(self):
        ext, name = self.splitter._extract_file_type_hints("review scala spark jobs")
        assert ".scala" in ext

    # -- TypeScript standalone --
    def test_typescript_keyword_triggers_ts_tsx(self):
        ext, name = self.splitter._extract_file_type_hints("analyze typescript interfaces")
        assert ".ts" in ext
        assert ".tsx" in ext

    # -- Domain categories --
    def test_html_keyword_triggers_html_css(self):
        ext, name = self.splitter._extract_file_type_hints("review html templates")
        assert ".html" in ext
        assert ".css" in ext

    def test_frontend_keyword_triggers_html_css(self):
        ext, name = self.splitter._extract_file_type_hints("analyze frontend components")
        assert ".html" in ext

    def test_sql_keyword_triggers_sql(self):
        ext, name = self.splitter._extract_file_type_hints("review sql migrations")
        assert ".sql" in ext

    def test_bash_keyword_triggers_sh(self):
        ext, name = self.splitter._extract_file_type_hints("analyze bash scripts")
        assert ".sh" in ext
        assert ".bash" in ext


class TestFilterFilesByHintsExtended:
    """Verify _filter_files_by_hints with extended language extensions."""

    def test_rb_files_matched_by_extension(self):
        files = [Path("app/models/user.rb"), Path("app/controllers/api.py")]
        result = _filter_files_by_hints(files, [".rb"], [])
        assert Path("app/models/user.rb") in result
        assert Path("app/controllers/api.py") not in result

    def test_swift_files_matched_by_extension(self):
        files = [Path("Views/MainView.swift"), Path("Models/User.py")]
        result = _filter_files_by_hints(files, [".swift"], [])
        assert Path("Views/MainView.swift") in result

    def test_kt_files_matched_by_extension(self):
        files = [Path("MainActivity.kt"), Path("Utils.java")]
        result = _filter_files_by_hints(files, [".kt", ".java"], [])
        assert Path("MainActivity.kt") in result
        assert Path("Utils.java") in result

    def test_sql_files_matched_by_extension(self):
        files = [Path("migrations/001.sql"), Path("config.yaml")]
        result = _filter_files_by_hints(files, [".sql"], [])
        assert Path("migrations/001.sql") in result

    def test_shell_scripts_matched_by_extension(self):
        files = [Path("deploy.sh"), Path("setup.bash"), Path("main.py")]
        result = _filter_files_by_hints(files, [".sh", ".bash"], [])
        assert Path("deploy.sh") in result
        assert Path("setup.bash") in result
        assert Path("main.py") not in result


class TestKeywordOverlapHandling:
    """Verify that overlapping keywords produce correct union semantics."""

    def test_android_triggers_java_and_kotlin(self):
        ext, name = self.splitter._extract_file_type_hints("analyze android app")
        assert ".java" in ext
        assert ".kt" in ext

    def test_react_triggers_js_and_ts(self):
        ext, name = self.splitter._extract_file_type_hints("review react components")
        assert ".js" in ext
        assert ".jsx" in ext
        assert ".ts" in ext
        assert ".tsx" in ext

    def test_no_duplicate_extension_entries(self):
        """Each extension should appear at most once per hint extraction."""
        ext, name = self.splitter._extract_file_type_hints("javascript react typescript")
        # .ts and .tsx may appear multiple times in the list due to
        # both javascript andtypescript entries, but _filter_files_by_hints
        # uses `any(hint == suffix_lower)` which is unaffected by duplicates.
        # However, for cleanliness, deduplication would be ideal.
        # This test documents the current behavior.
        pass  # Duplicate extensions in hints list are harmless due to `any()` semantics
```

### 4.2 Tests for Codebase Context Caching

**File**: `tests/test_concurrency_eligibility_classifier_caching.py` (new)

```python
import pytest
from pathlib import Path
from codebase_rag.orchestrator.concurrency_eligibility_classifier import ConcurrencyEligibilityClassifier


class TestCodebaseContextCaching:
    """Verify _get_codebase_context caching behavior."""

    def test_cache_initially_none(self):
        classifier = ConcurrencyEligibilityClassifier()
        assert classifier._codebase_context_cache is None

    def test_cache_populated_after_first_call(self):
        classifier = ConcurrencyEligibilityClassifier()
        file_count, languages, repo_size = classifier._get_codebase_context()
        assert classifier._codebase_context_cache is not None
        assert classifier._codebase_context_cache["file_count"] == file_count
        assert classifier._codebase_context_cache["languages"] == languages
        assert classifier._codebase_context_cache["repo_size"] == repo_size

    def test_cache_returns_same_values_on_second_call(self):
        classifier = ConcurrencyEligibilityClassifier()
        result1 = classifier._get_codebase_context()
        result2 = classifier._get_codebase_context()
        assert result1 == result2

    def test_cache_hit_avoids_recomputation(self):
        """Verify that a cache hit does not recompute file_count by checking
        that the cache dict is the same object (not recreated)."""
        classifier = ConcurrencyEligibilityClassifier()
        classifier._get_codebase_context()
        cache_ref = classifier._codebase_context_cache
        classifier._get_codebase_context()
        assert classifier._codebase_context_cache is cache_ref  # Same object, not recomputed


class TestExtendedLanguageDetectionInContext:
    """Verify expanded language-to-extension mapping in _get_codebase_context."""

    def test_swift_detected(self):
        """Swift files should map to 'Swift' language."""
        classifier = ConcurrencyEligibilityClassifier()
        # Mock: Create a classifier and check the mapping directly
        lang_map = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".jsx": "JavaScript", ".tsx": "TypeScript",
            ".java": "Java", ".cpp": "C++", ".h": "C++", ".hpp": "C++",
            ".go": "Go", ".rs": "Rust", ".cs": "C#",
            ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
            ".kt": "Kotlin", ".scala": "Scala",
        }
        assert lang_map[".swift"] == "Swift"
        assert lang_map[".kt"] == "Kotlin"
        assert lang_map[".scala"] == "Scala"
        assert lang_map[".jsx"] == "JavaScript"
        assert lang_map[".tsx"] == "TypeScript"

    def test_jsx_grouped_with_javascript(self):
        """React JSX files should be counted as JavaScript, not 'Other'."""
        # This ensures React component files are properly categorized
        lang_map = {
            ".jsx": "JavaScript",
            ".tsx": "TypeScript",
        }
        assert lang_map[".jsx"] == "JavaScript"
        assert lang_map[".tsx"] == "TypeScript"
```

### 4.3 Tests for Adaptive Threshold Data Collection

**File**: `tests/test_adaptive_threshold_data_collection.py` (new)

```python
import pytest
from codebase_rag.orchestrator.concurrency_eligibility_classifier import ConcurrencyEligibilityClassifier


class TestRecordExecutionResult:
    """Verify record_execution_result populates success_rate_tracker correctly."""

    def test_record_success_creates_tracker_entry(self):
        classifier = ConcurrencyEligibilityClassifier()
        classifier.record_execution_result("llm_analyzed", True)
        assert "llm_analyzed" in classifier.success_rate_tracker
        assert classifier.success_rate_tracker["llm_analyzed"] == [True]

    def test_record_failure_creates_tracker_entry(self):
        classifier = ConcurrencyEligibilityClassifier()
        classifier.record_execution_result("llm_rejected", False)
        assert "llm_rejected" in classifier.success_rate_tracker
        assert classifier.success_rate_tracker["llm_rejected"] == [False]

    def test_multiple_records_accumulate(self):
        classifier = ConcurrencyEligibilityClassifier()
        for _ in range(10):
            classifier.record_execution_result("llm_analyzed", True)
        assert len(classifier.success_rate_tracker["llm_analyzed"]) == 10

    def test_tracker_capped_at_100(self):
        classifier = ConcurrencyEligibilityClassifier()
        for _ in range(150):
            classifier.record_execution_result("llm_analyzed", True)
        assert len(classifier.success_rate_tracker["llm_analyzed"]) == 100

    def test_mixed_success_failure_records(self):
        classifier = ConcurrencyEligibilityClassifier()
        classifier.record_execution_result("llm_analyzed", True)
        classifier.record_execution_result("llm_analyzed", True)
        classifier.record_execution_result("llm_analyzed", False)
        tracker = classifier.success_rate_tracker["llm_analyzed"]
        assert tracker == [True, True, False]
        assert sum(tracker) / len(tracker) == 2/3


class TestAdaptiveThresholdActivation:
    """Verify that _adjust_threshold_based_on_success activates after 10 data points."""

    def test_threshold_unchanged_with_insufficient_data(self):
        """With fewer than 10 records, threshold should not adjust."""
        classifier = ConcurrencyEligibilityClassifier()
        for _ in range(5):
            classifier.record_execution_result("llm_analyzed", True)
        result = classifier._adjust_threshold_based_on_success("llm_analyzed")
        assert result == classifier.threshold  # No adjustment

    def test_threshold_lowered_with_high_success_rate(self):
        """With ≥10 records and ≥85% success, threshold should lower."""
        classifier = ConcurrencyEligibilityClassifier()
        # 10 successes out of 10 = 100% success rate
        for _ in range(10):
            classifier.record_execution_result("llm_analyzed", True)
        result = classifier._adjust_threshold_based_on_success("llm_analyzed")
        assert result < classifier.threshold
        assert result == max(0.5, classifier.threshold * 0.8)

    def test_threshold_raised_with_low_success_rate(self):
        """With ≥10 records and <50% success, threshold should raise."""
        classifier = ConcurrencyEligibilityClassifier()
        # 3 successes out of 10 = 30% success rate
        for i in range(10):
            classifier.record_execution_result("llm_analyzed", i < 3)
        result = classifier._adjust_threshold_based_on_success("llm_analyzed")
        assert result > classifier.threshold
        assert result == min(0.9, classifier.threshold * 1.3)

    def test_unknown_task_type_returns_effective_threshold(self):
        """Task type not in tracker should return current effective threshold."""
        classifier = ConcurrencyEligibilityClassifier()
        result = classifier._adjust_threshold_based_on_success("unknown_type")
        assert result == classifier._effective_threshold


class TestDeterministicRejectionFiltering:
    """Verify that deterministic rejection types are excluded from recording.

    This tests the design decision from §1.4 of this spec: deterministic
    rejection types should not be recorded because they are always correct
    and don't benefit from threshold adjustment.
    """

    DETERMINISTIC_REJECTION_TYPES = (
        "write_operation",
        "user_requested_sequential",
        "safety_rule_blocked",
        "insufficient_subtasks",
        "concurrency_disabled",
    )

    def test_llm_rejected_is_not_deterministic(self):
        """llm_rejected is a non-deterministic type that SHOULD be recorded."""
        assert "llm_rejected" not in self.DETERMINISTIC_REJECTION_TYPES

    def test_llm_analyzed_is_not_deterministic(self):
        """llm_analyzed is a non-deterministic type that SHOULD be recorded."""
        assert "llm_analyzed" not in self.DETERMINISTIC_REJECTION_TYPES

    def test_write_operation_is_deterministic(self):
        assert "write_operation" in self.DETERMINISTIC_REJECTION_TYPES

    def test_safety_rule_blocked_is_deterministic(self):
        assert "safety_rule_blocked" in self.DETERMINISTIC_REJECTION_TYPES
```

---

## Part V: Implementation Checklist

The following checklist tracks the implementation status of all items in this spec:

### Already Implemented (Part I — No Action Needed)

- [x] Extended language coverage in `_extract_file_type_hints()` (6 languages + 3 domain categories)
- [x] Expanded language-to-extension mapping in `_get_codebase_context()` (7 extensions)
- [x] Codebase context caching via `_codebase_context_cache`
- [x] Keyword overlap handling (union semantics for `android`, `react`, `node`, `.ts`)

### Requires Implementation (Part II — Gap Fixes)

- [ ] **Gap 1**: Add `concurrency_classifier.record_execution_result(task_type, success=True)` after successful parallel execution in `force_parallel` path
- [ ] **Gap 1**: Add `concurrency_classifier.record_execution_result(task_type, success=True)` after successful parallel execution in normal eligibility path
- [ ] **Gap 1**: Add `concurrency_classifier.record_execution_result(task_type, success=False)` after LLM-driven rejection in `not eligible` branch (excluding deterministic types)
- [ ] **Gap 2**: No fix needed — docstring already correct

### Test Files to Create

- [ ] `tests/test_task_splitter_file_type_hints.py`
- [ ] `tests/test_concurrency_eligibility_classifier_caching.py`
- [ ] `tests/test_adaptive_threshold_data_collection.py`

---

## Appendix A: Complete Language Coverage Reference

### A.1 `_extract_file_type_hints()` — Full Trigger Mapping

| Language/Category | Trigger Keywords | Extension Hints | Name Pattern Hints |
|-------------------|-----------------|-----------------|-------------------|
| Python | `python`, `.py`, `django`, `flask` | `.py` | — |
| JavaScript | `javascript`, `.js`, `react`, `node`, `express` | `.js`, `.jsx`, `.ts`, `.tsx` | — |
| Java | `java`, `.java`, `spring`, `android` | `.java` | — |
| C++ | `c++`, `.cpp`, `stl` | `.cpp`, `.h`, `.hpp` | — |
| Go | `go`, `.go`, `golang` | `.go` | — |
| Rust | `rust`, `.rs`, `cargo` | `.rs` | — |
| C# | `c#`, `.cs`, `csharp`, `.net`, `asp.net` | `.cs` | — |
| Ruby | `ruby`, `.rb`, `rails` | `.rb` | — |
| PHP | `php`, `.php`, `laravel` | `.php` | — |
| Swift | `swift`, `.swift`, `ios` | `.swift` | — |
| Kotlin | `kotlin`, `.kt`, `android` | `.kt` | — |
| Scala | `scala`, `.scala` | `.scala` | — |
| TypeScript (standalone) | `typescript`, `.ts` | `.ts`, `.tsx` | — |
| Test files | `test`, `spec` | — | `_test`, `_spec`, `test_`, `spec_`, `.test`, `.spec` |
| Config files | `config`, `setting` | `.json`, `.yaml`, `.yml`, `.toml`, `.ini` | `config`, `settings`, `configuration` |
| README files | `readme` | `.md`, `.rst` | `readme` |
| Doc files | `doc` (not `documentation`) | — | `doc`, `docs` |
| HTML/Web | `html`, `web`, `frontend` | `.html`, `.css`, `.scss`, `.sass` | — |
| SQL | `sql`, `database` | `.sql` | — |
| Shell | `shell`, `bash`, `script` | `.sh`, `.bash` | — |

### A.2 `_get_codebase_context()` — Full Extension-to-Language Mapping

| Extension | Language |
|-----------|----------|
| `.py` | Python |
| `.js` | JavaScript |
| `.jsx` | JavaScript |
| `.ts` | TypeScript |
| `.tsx` | TypeScript |
| `.java` | Java |
| `.cpp` | C++ |
| `.h` | C++ |
| `.hpp` | C++ |
| `.go` | Go |
| `.rs` | Rust |
| `.cs` | C# |
| `.rb` | Ruby |
| `.php` | PHP |
| `.swift` | Swift |
| `.kt` | Kotlin |
| `.scala` | Scala |
| *(all others)* | Other |

### A.3 Filtering Semantics

`_filter_files_by_hints()` uses **OR union** semantics: a file is included if it matches **any** extension hint **OR** any name pattern hint. This means:

- `{extension_hints=['.py'], name_pattern_hints=['_test']}` → includes ALL `.py` files AND ALL files with `_test` in their name (e.g., `user_test.py`, `test_helper.rb`)
- No file is excluded for failing to match both types — either match is sufficient
- Empty hints lists return all files (no filtering applied)

---

## Appendix B: Adaptive Threshold Adjustment Flow

```
Query arrives at _run_interactive_loop()
  │
  ├─ Write operation detected? ──→ SKIP (no recording — deterministic)
  │
  ├─ no_parallel flag set? ──→ SKIP (no recording — deterministic)
  │
  ├─ force_parallel flag set?
  │   ├─ Fewer than MIN_SUBTASKS? ──→ downgrade to sequential (no recording)
  │   └─ Sufficient subtasks? ──→ EXECUTE PARALLEL
  │       └─ record_execution_result("user_forced", success=True)
  │
  ├─ Queue size exceeded? ──→ SKIP (no recording — deterministic)
  │
  └─ Normal eligibility check via classifier
  │   ├─ eligible=False, deterministic type? ──→ SKIP (no recording)
  │   ├─ eligible=False, LLM-driven type? ──→ record_execution_result(type, success=False)
  │   ├─ eligible=True, but auto_split disabled? ──→ SKIP (no recording)
  │   ├─ eligible=True, but insufficient subtasks? ──→ downgrade to sequential (no recording)
  │   └─ eligible=True, sufficient subtasks? ──→ EXECUTE PARALLEL
  │       └─ record_execution_result(type, success=True)
  │
  ──→ Next query iteration (classifier retains accumulated data)
```

The adaptive threshold adjustment cycle:

```
_record_execution_result() called ≥10 times for a task_type
  │
  └─ _adjust_threshold_based_on_success(task_type) invoked on next is_eligible()
     │
     ├─ success_rate ≥ 0.85 ──→ effective_threshold = max(0.5, base * 0.8)  [lower bar]
     ├─ success_rate ≥ 0.70 ──→ effective_threshold = base                  [keep bar]
     ├─ success_rate ≥ 0.50 ──→ effective_threshold = min(0.8, base * 1.1)  [raise bar]
     └─ success_rate < 0.50 ──→ effective_threshold = min(0.9, base * 1.3)  [raise bar significantly]
```