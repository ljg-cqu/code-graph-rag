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

### Fix 3: Improve Model Validation with Actual API Check

Add a lightweight API call to truly validate model availability:

```python
def _validate_model_availability(self, model_config: ModelConfig) -> bool:
    """
    Check if the specified model is available before spawning sub-agents.
    Performs a lightweight API call to verify availability.
    Returns True if model is available, False otherwise.
    """
    try:
        provider = get_provider_from_config(model_config)
        model = provider.create_model(model_config.model_id)

        # Perform a minimal test request to verify the model actually works
        # This catches 404 errors that create_model() doesn't
        if hasattr(model, 'validate_availability'):
            return model.validate_availability()

        # Fallback: try a minimal request if no validate_availability method
        # Use a simple ping-style request if the provider supports it
        if hasattr(provider, 'ping_model'):
            return provider.ping_model(model_config.model_id)

        # Last resort: assume available if we got this far
        return True

    except Exception as e:
        logger.warning(
            f"Model '{model_config.model_id}' is not available: {e}"
        )
        return False
```

### Fix 4: Add Failed Model Tracking to Avoid Repeated Attempts

Track failed models to avoid retrying the same model multiple times:

```python
class SubagentOrchestrator:
    # ... existing attributes ...
    _failed_models: set[str] = set()  # Track models that have failed

    def _get_fallback_model_config(self, original_config: ModelConfig) -> ModelConfig | None:
        original_model_id = original_config.model_id
        fallback_chain = self.MODEL_FALLBACK_CHAIN.get(original_model_id, [])

        if not fallback_chain:
            fallback_chain = [
                m for m in ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"]
                if m != original_model_id
            ]

        # Filter out models that have already failed
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

        # Also mark the original as failed
        self._failed_models.add(original_model_id)
        return None
```

### Fix 5: Clear Error Propagation When All Models Fail

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
    if not self.workers or not any(w.agent for w in self.workers):
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

## Implementation Plan

**File:** `codebase_rag/orchestrator/subagent_orchestrator.py`

1. **Line ~289**: Update `MODEL_FALLBACK_CHAIN` with complete entries for all models

2. **Line ~296** (in `__init__`): Add `_failed_models: set[str] = set()`

3. **Line ~361-384**: Replace `_get_fallback_model_config()` with improved version

4. **Line ~345-359**: Update `_validate_model_availability()` for actual API validation

5. **Line ~450**: Add early check for no valid agents in `execute_tasks()`

## Testing Strategy

### Unit Tests

```python
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

    # initialize_agents should log error and not create workers
    with patch.object(settings, 'active_worker_llms', []):
        orchestrator.initialize_agents()

        # Should have no valid workers
        assert len(orchestrator.workers) == 0 or all(w.agent is None for w in orchestrator.workers)
```

### Integration Test

```python
async def test_graceful_degradation_with_unavailable_models():
    """Test that parallel execution fails gracefully when all models 404."""
    orchestrator = SubagentOrchestrator(worker_count=2)

    # Mock all models returning 404
    with patch('codebase_rag.providers.get_provider_from_config') as mock_provider:
        mock_provider.return_value.create_model.side_effect = Exception("404: Not found")

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

- [ ] Fallback never attempts to use the same model that just failed
- [ ] Complete fallback chain for all Claude model variants
- [ ] Failed models are tracked and not retried within same session
- [ ] Clear error message when all models are unavailable
- [ ] Model validation performs actual API check (not just object creation)
- [ ] Logs clearly show which model is being tried and why
- [ ] Parallel execution either succeeds with valid model or fails fast with clear error

## Related Files

- `codebase_rag/orchestrator/subagent_orchestrator.py` - Main file to modify
- `codebase_rag/providers/__init__.py` - May need `ping_model` method
- `codebase_rag/config.py` - Model configuration

## Migration Notes

This fix is backward compatible. The only behavior change is:
1. Better fallback logic (no self-referential fallbacks)
2. Clearer error messages when models fail
3. No functional changes to the API or configuration

## Out of Scope

- Adding new LLM providers
- Changing the retry count or timeout values
- Modifying the sub-agent prompt or tool set
