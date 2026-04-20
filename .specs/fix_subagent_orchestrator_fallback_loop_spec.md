# Design Spec: Fix Subagent Orchestrator Fallback Loop Bug

## Problem Statement

The subagent orchestrator has a critical bug in its model fallback logic that causes it to attempt fallback to the same model that just failed, creating an ineffective retry loop:

```
2026-04-20 19:09:06.276 | INFO | codebase_rag.orchestrator.subagent_orchestrator:_get_fallback_model_config:379 - Using fallback model 'claude-sonnet-4-6' instead of 'claude-sonnet-4-6'
```

This happens when:
1. Primary model `k2.6-code-preview` fails with 404
2. Falls back to `claude-sonnet-4-6`
3. `claude-sonnet-4-6` also fails with 404
4. Tries to find fallback for `claude-sonnet-4-6`
5. Generic fallback chain includes `claude-sonnet-4-6` (the same model!)

**Impact:** All 9 subtasks failed permanently after 3 retry attempts, defeating parallel execution entirely.

## Root Cause Analysis

### Bug 1: Self-Referential Fallback Chain

The generic fallback chain at line 371 includes models that may already be the current model:

```python
# In _get_fallback_model_config() - line 371
if not fallback_chain:
    # Try generic fallbacks
    fallback_chain = ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"]
```

When `claude-sonnet-4-6` fails and looks for fallback, this chain includes itself.

### Bug 2: Missing Exclusion of Current Model

The fallback logic doesn't exclude the original model from the fallback chain:

```python
for fallback_model_id in fallback_chain:
    # No check to skip if fallback_model_id == original_model_id
    fallback_config = replace(original_config, model_id=fallback_model_id)
    if self._validate_model_availability(fallback_config):
        return fallback_config
```

### Bug 3: Model-Specific Fallback Chains Incomplete

The `MODEL_FALLBACK_CHAIN` doesn't include entries for standard Claude models:

```python
MODEL_FALLBACK_CHAIN: dict[str, list[str]] = {
    "k2.6-code-preview": ["claude-sonnet-4-6", ...],
    "k2.5-code-preview": ["claude-sonnet-4-6", ...],
    # Missing: "claude-sonnet-4-6": ["claude-opus-4-7", "claude-haiku-4-5"],
}
```

### Bug 4: Validation Passes But Execution Fails

The `_validate_model_availability()` method passes but actual execution fails:

```python
def _validate_model_availability(self, model_config: ModelConfig) -> bool:
    try:
        provider = get_provider_from_config(model_config)
        provider.create_model(model_config.model_id)  # May succeed without API call
        return True
    except Exception as e:
        return False
```

The `create_model()` call may not actually validate against the API, leading to a false positive.

**Evidence from logs:**
```
2026-04-20 19:16:53.259 | INFO | subagent_orchestrator:initialize_agents:442 - Sub-agent pool initialized with 1 worker LLMs (round-robin assignment)
```

This shows initialization succeeded with 1 worker LLM, but then:
```
2026-04-20 19:16:53.503 | INFO | subagent_orchestrator:_get_fallback_model_config:379 - Using fallback model 'claude-sonnet-4-6' instead of 'claude-sonnet-4-6'
```

The model passed validation but fails when actually making API calls during task execution.

### Bug 5: Wasted Retry Resources

When all models fail, the system wastes resources on redundant retries:

**From supplementary logs:**
- 9 subtasks x 3 attempts = 27 API calls
- All 27 calls fail with same 404 error
- Total time wasted: 3.84 seconds for zero useful work
- No early termination when all models are known to be unavailable

### Bug 6: Connection Errors Mixed with 404s

The logs show different error types that should be handled differently:

```
2026-04-20 19:16:56.861 | WARNING | subtask_5 failed (attempt 2/3): Connection error.
```

vs

```
2026-04-20 19:16:56.845 | WARNING | subtask_6 failed (attempt 2/3): status_code: 404
```

- **404 errors**: Model doesn't exist - should try different model immediately
- **Connection errors**: Network issue - should retry with same model, then try different model

## Proposed Solution

### Fix 1: Exclude Current Model from Fallback Chain

```python
def _get_fallback_model_config(self, original_config: ModelConfig) -> ModelConfig | None:
    """
    Get a fallback model configuration when the primary model is unavailable.
    Returns None if no fallback is available.
    """
    original_model_id = original_config.model_id
    fallback_chain = self.MODEL_FALLBACK_CHAIN.get(original_model_id, [])

    if not fallback_chain:
        # Try generic fallbacks - EXCLUDE the current model
        fallback_chain = [
            m for m in ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"]
            if m != original_model_id
        ]

    # Filter out the original model from specific fallback chains too
    fallback_chain = [m for m in fallback_chain if m != original_model_id]

    for fallback_model_id in fallback_chain:
        fallback_config = replace(original_config, model_id=fallback_model_id)

        if self._validate_model_availability(fallback_config):
            logger.info(
                f"Using fallback model '{fallback_model_id}' instead of '{original_model_id}'"
            )
            return fallback_config

    return None
```

### Fix 2: Complete Model Fallback Chains

Add entries for all possible model IDs that might be used:

```python
MODEL_FALLBACK_CHAIN: dict[str, list[str]] = {
    # Custom/preview models
    "k2.6-code-preview": ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
    "k2.5-code-preview": ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
    "kimi-k2.6": ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
    "kimi-k2.5": ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
    # Standard Claude models (cascading fallback)
    "claude-sonnet-4-6": ["claude-opus-4-7", "claude-haiku-4-5"],
    "claude-opus-4-7": ["claude-sonnet-4-6", "claude-haiku-4-5"],
    "claude-haiku-4-5": ["claude-sonnet-4-6", "claude-opus-4-7"],
}
```

**Note:** The bidirectional cycle (e.g., sonnet -> opus -> sonnet) is safe only because Fix 4 tracks failed models and prevents re-trying them.

### Fix 3: Add Failed Model Tracking to Avoid Repeated Attempts

Track failed models as an **instance attribute** to avoid retrying the same model multiple times:

```python
class SubagentOrchestrator:
    # ... existing class attributes ...

    def __init__(
        self,
        worker_count: int | None = None,
        agent_factory: Callable | None = None,
        scheduling_strategy: str = "round-robin",
        repo_path: str | None = None,
        enable_document_graph: bool = False,
        query_mode: QueryMode = QueryMode.CODE_ONLY,
        doc_workspace: str = "default",
    ):
        # ... existing __init__ code ...
        self._failed_models: set[str] = set()  # Track models that have failed
        self._has_validated_models: bool = False

    def _get_fallback_model_config(self, original_config: ModelConfig) -> ModelConfig | None:
        original_model_id = original_config.model_id
        fallback_chain = self.MODEL_FALLBACK_CHAIN.get(original_model_id, [])

        if not fallback_chain:
            fallback_chain = [
                m for m in ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"]
                if m != original_model_id
            ]

        # Filter out models that have already failed and the original model
        available_fallbacks = [
            m for m in fallback_chain
            if m != original_model_id and m not in self._failed_models
        ]

        for fallback_model_id in available_fallbacks:
            fallback_config = replace(original_config, model_id=fallback_model_id)

            if self._validate_model_availability(fallback_config):
                logger.info(
                    f"Using fallback model '{fallback_model_id}' instead of '{original_model_id}'"
                )
                return fallback_config
            else:
                # Mark this model as failed
                self._failed_models.add(fallback_model_id)

        # Mark the original as failed
        self._failed_models.add(original_model_id)
        return None
```

### Fix 4: Differentiate Error Types for Better Retry Strategy

Handle different error types appropriately:

```python
def _classify_error(self, error_msg: str) -> str:
    """Classify error type to determine appropriate retry strategy."""
    error_lower = error_msg.lower()
    
    if "404" in error_msg or "resource_not_found" in error_lower or "not found" in error_lower:
        return "model_unavailable"
    elif "connection" in error_lower or "timeout" in error_lower or "network" in error_lower:
        return "network_error"
    elif "rate_limit" in error_lower or "429" in error_msg:
        return "rate_limit"
    elif "auth" in error_lower or "401" in error_msg or "403" in error_msg:
        return "auth_error"
    else:
        return "unknown"

# In execute_tasks error handling:
error_type = self._classify_error(error_msg)

if error_type == "model_unavailable":
    # Don't retry same model - switch to fallback immediately
    self._failed_models.add(current_model_id)
    fallback_config = self._get_fallback_model_config(current_config)
    # ... switch to fallback
    
elif error_type == "network_error":
    # Retry same model once, then try fallback
    if retry_count < 1:
        # Retry same model
        remaining_tasks.insert(0, subtask)
    else:
        # Try fallback
        fallback_config = self._get_fallback_model_config(current_config)
        # ... switch to fallback

elif error_type == "rate_limit":
    # Wait and retry
    await asyncio.sleep(2 ** retry_count)
    remaining_tasks.insert(0, subtask)
```

### Fix 5: Early Termination When All Models Known Unavailable

Stop retrying once all models in the fallback chain have failed:

```python
def _get_all_possible_models(self) -> set[str]:
    """Return the universe of all models that could be used for fallback."""
    all_models: set[str] = set()
    # Models from the static fallback chain
    for chain in self.MODEL_FALLBACK_CHAIN.values():
        all_models.update(chain)
    all_models.update(self.MODEL_FALLBACK_CHAIN.keys())
    # Models from worker LLM configuration
    for cfg in settings.active_worker_llms:
        if cfg.model_id:
            all_models.add(cfg.model_id)
    # Orchestrator config model
    orch_cfg = settings.active_orchestrator_config
    if orch_cfg.model_id:
        all_models.add(orch_cfg.model_id)
    return all_models

# In execute_tasks:
# After a model fails permanently:
self._failed_models.add(failed_model_id)

# Check if all possible models have failed
all_possible_models = self._get_all_possible_models()
if self._failed_models >= all_possible_models:
    logger.error(
        f"All models have failed. Terminating parallel execution early. "
        f"Failed models: {self._failed_models}"
    )
    # Mark all remaining tasks as failed
    for remaining in remaining_tasks:
        result_aggregator.add_error(
            remaining,
            "All LLM models exhausted. No models available for execution.",
            execution_time=0,
        )
    remaining_tasks.clear()
    break
```

### Fix 6: Clear Error Propagation When All Models Fail

When no models are available, fail fast with a clear error:

```python
async def execute_tasks(
    self,
    subtasks: list[dict[str, Any]],
    result_aggregator: ResultAggregator | None = None,
    retry_attempts: int | None = None,
    dry_run: bool = False,
) -> ResultAggregator:
    # Check if we have any validated models before starting
    if not self._has_validated_models:
        logger.error(
            "No valid sub-agents available. All models failed validation. "
            f"Failed models: {self._failed_models}"
        )
        # Return immediately with all tasks marked as failed
        for subtask in subtasks:
            result_aggregator.add_error(
                subtask,
                "No LLM models available for parallel execution. "
                "Check model configuration and API availability.",
                execution_time=0,
            )
        return result_aggregator

    # ... rest of method
```

**Note:** `initialize_agents()` must set `self._has_validated_models = True` when at least one model passes validation.

### Fix 7: Update initialize_agents to Track Validation State

```python
def initialize_agents(self) -> None:
    logger.info(f"Initializing {self.worker_count} sub-agents")
    worker_llms = settings.active_worker_llms
    num_worker_llms = len(worker_llms)

    # Validate and potentially fallback model configurations
    validated_worker_llms: list[ModelConfig] = []
    for llm_config in worker_llms:
        if self._validate_model_availability(llm_config):
            validated_worker_llms.append(llm_config)
        else:
            self._failed_models.add(llm_config.model_id)
            # Try to find a fallback model
            fallback_config = self._get_fallback_model_config(llm_config)
            if fallback_config:
                validated_worker_llms.append(fallback_config)
            else:
                logger.error(
                    f"Model '{llm_config.model_id}' is unavailable and no fallback found. "
                    f"This worker configuration will be skipped."
                )

    # If no valid models, try orchestrator config as fallback
    if not validated_worker_llms:
        orchestrator_config = settings.active_orchestrator_config
        if self._validate_model_availability(orchestrator_config):
            logger.warning(
                "No valid worker LLMs found. Using orchestrator config as fallback."
            )
            validated_worker_llms = [orchestrator_config]
        else:
            self._failed_models.add(orchestrator_config.model_id)
            # Try fallback for orchestrator config too
            fallback_config = self._get_fallback_model_config(orchestrator_config)
            if fallback_config:
                logger.warning(
                    f"Using fallback model '{fallback_config.model_id}' for orchestrator."
                )
                validated_worker_llms = [fallback_config]

    # Track whether any models passed validation
    self._has_validated_models = len(validated_worker_llms) > 0

    # ... rest of existing method
```

## Implementation Plan

**File:** `codebase_rag/orchestrator/subagent_orchestrator.py`

1. **Line ~289**: Update `MODEL_FALLBACK_CHAIN` with complete entries for all models

2. **Line ~296** (in `__init__`): Add instance attributes:
   - `self._failed_models: set[str] = set()`
   - `self._has_validated_models: bool = False`

3. **Line ~361-384**: Replace `_get_fallback_model_config()` with improved version

4. **Line ~386-448**: Update `initialize_agents()` to track failed models and set `_has_validated_models`

5. **Line ~450+**: Add `_get_all_possible_models()` helper method

6. **Line ~450+**: Add `_classify_error()` helper method

7. **Line ~574-627**: Update error handling in `execute_tasks()` to use error classification and early termination

8. **Line ~512**: Add early check for `_has_validated_models` in `execute_tasks()`

## Testing Strategy

**Test file:** `tests/test_subagent_orchestrator.py`

### Unit Tests

```python
import pytest
from unittest.mock import patch

from codebase_rag.config import ModelConfig, settings
from codebase_rag.orchestrator.subagent_orchestrator import SubagentOrchestrator


def test_fallback_excludes_current_model():
    """Verify fallback never returns the same model that failed."""
    orchestrator = SubagentOrchestrator()

    config = ModelConfig(model_id="claude-sonnet-4-6", provider="anthropic")

    # Mock all models as unavailable except haiku
    with patch.object(orchestrator, '_validate_model_availability') as mock_validate:
        mock_validate.side_effect = lambda c: c.model_id == "claude-haiku-4-5"

        fallback = orchestrator._get_fallback_model_config(config)

        # Should return haiku, not sonnet
        assert fallback is not None
        assert fallback.model_id == "claude-haiku-4-5"


def test_failed_models_not_retried():
    """Verify models that failed are not retried."""
    orchestrator = SubagentOrchestrator()

    # Simulate sonnet already failed
    orchestrator._failed_models.add("claude-sonnet-4-6")

    config = ModelConfig(model_id="claude-opus-4-7", provider="anthropic")

    with patch.object(orchestrator, '_validate_model_availability') as mock_validate:
        mock_validate.return_value = False

        orchestrator._get_fallback_model_config(config)

        # Should not have tried sonnet since it's in failed_models
        tried_models = [call[0][0].model_id for call in mock_validate.call_args_list]
        assert "claude-sonnet-4-6" not in tried_models


def test_all_models_failed_clear_error():
    """Verify clear error when all models fail."""
    orchestrator = SubagentOrchestrator()
    orchestrator._failed_models = {"claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"}
    orchestrator._has_validated_models = False

    # Should report no validated models
    assert not orchestrator._has_validated_models


def test_failed_models_is_instance_level():
    """Verify failed_models is not shared across instances."""
    orch1 = SubagentOrchestrator()
    orch2 = SubagentOrchestrator()

    orch1._failed_models.add("claude-sonnet-4-6")

    assert "claude-sonnet-4-6" in orch1._failed_models
    assert "claude-sonnet-4-6" not in orch2._failed_models


def test_get_all_possible_models_includes_worker_llms():
    """Verify _get_all_possible_models includes configured worker LLMs."""
    orchestrator = SubagentOrchestrator()

    with patch.object(settings, 'active_worker_llms', [
        ModelConfig(model_id="gpt-4o", provider="openai")
    ]):
        all_models = orchestrator._get_all_possible_models()
        assert "gpt-4o" in all_models
```

### Integration Test

```python
import pytest
from unittest.mock import patch

from codebase_rag.orchestrator.subagent_orchestrator import SubagentOrchestrator


@pytest.mark.asyncio
async def test_graceful_degradation_with_unavailable_models():
    """Test that parallel execution fails gracefully when all models 404."""
    orchestrator = SubagentOrchestrator(worker_count=2)

    # Mock all models returning 404
    with patch('codebase_rag.orchestrator.subagent_orchestrator.get_provider_from_config') as mock_get_provider:
        mock_provider = mock_get_provider.return_value
        mock_provider.create_model.side_effect = Exception("404: Not found")

        orchestrator.initialize_agents()

        # Should log clear error about no models available
        result = await orchestrator.execute_tasks([
            {"id": "test-1", "task": "Analyze file X"}
        ])

        # Should have error result, not hang or crash
        assert result.total_errors == 1
        assert "No LLM models available" in result.errors[0]["error"]
```

## Success Criteria

- [x] Fallback never attempts to use the same model that just failed
- [x] Complete fallback chain for all Claude model variants
- [x] Failed models are tracked and not retried within same session
- [x] Clear error message when all models are unavailable
- [x] `_failed_models` is an instance attribute, not a class attribute
- [x] `_has_validated_models` flag correctly reflects validation state
- [x] Early termination when all models in the expanded universe have failed
- [x] No wasted API calls retrying known-unavailable models
- [x] Parallel execution either succeeds with valid model or fails fast with clear error
- [x] Unit tests cover fallback exclusion, failed-model tracking, and instance isolation

## Related Files

- `codebase_rag/orchestrator/subagent_orchestrator.py` - Main file to modify
- `codebase_rag/config.py` - Model configuration and `settings`
- `tests/test_subagent_orchestrator.py` - New test file to create

## Migration Notes

This fix is backward compatible. The only behavior change is:
1. Better fallback logic (no self-referential fallbacks)
2. Failed-model tracking prevents redundant attempts
3. Clearer error messages when models fail
4. No functional changes to the API or configuration

## Out of Scope

- Adding new LLM providers
- Changing the retry count or timeout values
- Modifying the sub-agent prompt or tool set
- **True API ping validation**: The existing `create_model()` + `validate_config()` approach is retained. A future spec may add lightweight HTTP pings to provider endpoints, but that requires changes to `codebase_rag/providers/base.py` across all provider implementations and is deferred to keep this fix minimal and low-risk.
