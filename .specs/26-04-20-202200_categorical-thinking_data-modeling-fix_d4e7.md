# Design Spec: Categorical Thinking Knowledge Base — LLM Fail-Fast Implementation

**Spec ID:** 26-04-20-202200_categorical-thinking_data-modeling-fix_d4e7
**Target System:** Categorical Thinking Knowledge Base (v1.1.0)
**Generated:** 2026-04-20T20:22:00Z
**Updated:** 2026-04-20T22:45:00Z
**Priority:** P0 (Critical) — All 9 parallel ingestion subtasks failed
**Status:** ✅ COMPLETE — Fail-Fast Model Implemented

---

## 1. Executive Summary

The parallel ingestion of 9 data artifacts into the knowledge graph failed with 100% error rate (9/9). All failures share the same root cause: **HTTP 404 resource_not_found errors** from the LLM inference layer due to incorrect model IDs.

**Resolution:** Removed all LLM fallback logic and implemented fail-fast behavior. The system now uses only explicitly configured models. If the configured model is unavailable, the system fails immediately with a clear error message rather than silently degrading or cycling through fallback models.

---

## 2. Issue

### 2.1 P0 — Model Fallback Logic Causing 100% Failure Rate

| Field | Value |
|-------|-------|
| **ID** | BUG-001 |
| **Severity** | Blocking |
| **Category** | Configuration / Model Routing |
| **Symptom** | All 9 subtasks fail with `status_code: 404`, `type: resource_not_found_error`. The fallback chain cycled through invalid model IDs. |
| **Root Cause** | The `MODEL_FALLBACK_CHAIN` contained model IDs that don't exist in Anthropic's registry. Pydantic-ai accepts any string as `model_id` and only fails during actual API calls, so the fallback logic silently cycled through broken model IDs. |
| **Impact** | Zero data ingested into graph; 100% failure rate; confusing logs showing multiple fallback attempts |

**Solution:** Remove fallback logic entirely. Use only explicitly configured models. Fail fast with clear error messages.

---

## 3. Implementation

### 3.1 Changes Made

| Component | Action |
|-----------|--------|
| `MODEL_FALLBACK_CHAIN` class variable | Removed |
| `_failed_models` instance variable | Removed |
| `_has_validated_models` instance variable | Removed |
| `_validate_model_availability()` method | Renamed to `_validate_model_config()`, changed to raise `RuntimeError` |
| `_get_fallback_model_config()` method | Removed |
| `_validate_model_can_respond()` method | Removed |
| `_get_all_possible_models()` method | Removed |
| `initialize_agents()` method | Simplified to validate all configs upfront and fail fast |
| `execute_tasks()` method | Simplified to only retry transient errors (network/rate_limit) |

### 3.2 New `_validate_model_config()` Method

```python
def _validate_model_config(self, model_config: ModelConfig) -> None:
    """
    Validate that the model configuration can create a model.
    Raises RuntimeError if validation fails (fail-fast behavior).
    """
    try:
        provider = get_provider_from_config(model_config)
        provider.create_model(model_config.model_id)
    except Exception as e:
        raise RuntimeError(
            f"Model '{model_config.model_id}' from provider '{model_config.provider}' is not available: {e}"
        ) from e
```

### 3.3 Simplified `initialize_agents()` Method

```python
def initialize_agents(self) -> None:
    """Initialize sub-agents with configured LLMs. Fails fast if any model is unavailable."""
    logger.info(f"Initializing {self.worker_count} sub-agents")
    worker_llms = settings.active_worker_llms

    # Determine which LLM configs to use
    llm_configs: list[ModelConfig]
    if worker_llms:
        llm_configs = worker_llms
    else:
        llm_configs = [settings.active_orchestrator_config]

    # Validate all configs upfront - fail fast
    for llm_config in llm_configs:
        self._validate_model_config(llm_config)

    num_llms = len(llm_configs)

    # Adjust worker count if fewer LLMs than workers
    if num_llms == 1 and self.worker_count > 1:
        logger.warning(
            f"Only 1 LLM config available. Reducing worker_count from "
            f"{self.worker_count} to 1 to avoid false parallelism contention."
        )
        self.worker_count = 1

    # ... create workers with round-robin LLM assignment ...
```

### 3.4 Simplified Error Handling in `execute_tasks()`

```python
except Exception as e:
    error_msg = str(e)
    retry_count = retry_counts.get(subtask["id"], 0)
    error_type = self._classify_error(error_msg)

    # Only retry for transient network/rate-limit errors
    if error_type in ("network_error", "rate_limit") and retry_count < retry_attempts:
        retry_counts[subtask["id"]] = retry_count + 1
        if error_type == "rate_limit":
            time.sleep(2 ** retry_count)
        logger.warning(f"Subtask {subtask['id']} failed with {error_type}. Retrying...")
        remaining_tasks.insert(0, subtask)
    else:
        # Fail fast - no model fallback
        result_aggregator.add_error(subtask, error_msg, ...)
        logger.error(f"Subtask {subtask['id']} failed: {error_msg}")
```

---

## 4. Error Classification

| Error Type | Pattern | Retry? |
|------------|---------|--------|
| `model_unavailable` | 404, resource_not_found, not found | No |
| `auth_error` | 401, 403, auth | No |
| `network_error` | connection, timeout, network | Yes |
| `rate_limit` | 429, rate_limit, rate limit | Yes |
| `unknown` | (anything else) | No |

---

## 5. Affected Files

| File | Changes |
|------|---------|
| `codebase_rag/orchestrator/subagent_orchestrator.py` | Removed fallback logic, implemented fail-fast |
| `tests/test_subagent_orchestrator.py` | Rewrote tests for fail-fast behavior (15 tests) |

---

## 6. Validation

| Criterion | Result |
|-----------|--------|
| All tests pass | ✅ 119 tests pass |
| Fail-fast on invalid model | ✅ Raises `RuntimeError` immediately |
| No fallback attempts | ✅ Single error message, no fallback logs |
| Network errors retry | ✅ Up to configured limit |
| Auth/model errors fail immediately | ✅ No retry |

---

*Spec generated: 2026-04-20T20:22:00Z*
*Updated with fail-fast implementation: 2026-04-20T22:45:00Z*
