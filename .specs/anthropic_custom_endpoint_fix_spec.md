# Anthropic Provider Custom Endpoint Support Specification

## Problem Statement

When users configure `ORCHESTRATOR_PROVIDER=anthropic` with a custom endpoint (e.g., Baidu Qianfan's Anthropic-compatible API), the system fails with:

```
httpcore.LocalProtocolError: Illegal header value b'Bearer '
```

The root cause is a combination of issues:

1. **`AnthropicProvider` doesn't support custom endpoints**: The `endpoint` parameter is ignored entirely
2. **Empty `ANTHROPIC_AUTH_TOKEN` environment variable**: Interferes with authentication
3. **Proxy interference**: Requests to `api.anthropic.com` are intercepted by user's HTTP proxy

### Current Issues

#### Issue 1: `AnthropicProvider` Ignores Custom Endpoints

```python
class AnthropicProvider(ModelProvider):
    __slots__ = ("api_key",)  # Missing 'endpoint'

    def __init__(
        self,
        api_key: str | None = None,
        **kwargs: str | int | None,  # endpoint is lost here
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = _resolve_api_key(api_key, cs.ENV_ANTHROPIC_API_KEY)

    def create_model(self, model_id: str, **kwargs) -> AnthropicModel:
        provider = PydanticAnthropicProvider(api_key=self.api_key)
        #                                              ^^^^^^^^^^^^^^^^
        # base_url is never passed!
        return AnthropicModel(model_id, provider=provider)
```

The user's configuration:
```env
ORCHESTRATOR_PROVIDER=anthropic
ORCHESTRATOR_MODEL=GLM-5
ORCHESTRATOR_API_KEY=bce-v3/ALTAKSP-ZQ4hO485izmiCEGl3IUnH/...
ORCHESTRATOR_ENDPOINT=https://qianfan.baidubce.com/anthropic/coding  # IGNORED!
```

#### Issue 2: Empty/Whitespace API Keys Not Rejected

The `_resolve_api_key` function treats empty strings and whitespace-only strings as valid API keys, which can cause authentication errors when passed to the underlying provider.

#### Issue 3: Proxy Interception Without Custom Endpoint

Since the custom endpoint is ignored, requests go to `api.anthropic.com`, which the user's proxy (`http://127.0.0.1:7890`) intercepts and routes through `token-plan-cn.xiaomimimo.com`, causing the observed error.

## Design Goals

1. **Support custom Anthropic-compatible endpoints**: Allow users to configure `ORCHESTRATOR_ENDPOINT` for Anthropic providers
2. **Handle empty/whitespace API keys gracefully**: Validate and reject empty credentials early
3. **Maintain backward compatibility**: Existing configurations without custom endpoints continue to work
4. **Follow existing patterns**: Match the `OpenAIProvider` implementation for consistency

## Solution Architecture

### Component Changes

#### 1. Update `AnthropicProvider` in `providers/base.py`

Add `endpoint` support to match `OpenAIProvider`:

```python
class AnthropicProvider(ModelProvider):
    __slots__ = ("api_key", "endpoint")  # Add 'endpoint'

    MODEL_CONTEXT_WINDOWS = {
        "claude-3-5-sonnet*": 200000,
        "claude-3-opus*": 200000,
        "claude-3-sonnet*": 200000,
        "claude-3-haiku*": 200000,
        "claude-2.1*": 200000,
        "claude-2.0*": 100000,
    }

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,  # Add parameter
        **kwargs: str | int | None,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = _resolve_api_key(api_key, cs.ENV_ANTHROPIC_API_KEY)
        self.endpoint = endpoint or os.environ.get(cs.ENV_ANTHROPIC_BASE_URL)

    @property
    def provider_name(self) -> cs.Provider:
        return cs.Provider.ANTHROPIC

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError(ex.ANTHROPIC_NO_KEY)
        # Validate API key is not empty/whitespace
        if not self.api_key.strip():
            raise ValueError("Anthropic API key cannot be empty or whitespace")

    def create_model(self, model_id: str, **kwargs: str | int | None) -> AnthropicModel:
        self.validate_config()
        assert self.api_key is not None
        provider = PydanticAnthropicProvider(
            api_key=self.api_key,
            base_url=self.endpoint,  # Pass custom endpoint
        )
        return AnthropicModel(model_id, provider=provider)
```

#### 2. Add `ENV_ANTHROPIC_ENDPOINT` Constant in `constants.py`

```python
ENV_ANTHROPIC_ENDPOINT = "ANTHROPIC_ENDPOINT"
```

#### 3. Add `ANTHROPIC_NO_ENDPOINT` Error Message (Optional)

For validation error messages if needed.

#### 4. Update `_resolve_api_key` to Handle Whitespace (All Providers)

**Note**: This change affects all providers for consistency and safety.

```python
def _resolve_api_key(api_key: str | None, env_var: str) -> str | None:
    # Handle whitespace-only API keys
    if api_key and api_key.strip() and api_key != cs.DEFAULT_API_KEY:
        return api_key
    env_key = os.environ.get(env_var)
    # Also check env_key is not whitespace-only
    if env_key and env_key.strip():
        return env_key
    return None
```

### Environment Variable Precedence

For Anthropic endpoint resolution:
1. `ORCHESTRATOR_ENDPOINT` / `CYPHER_ENDPOINT` (role-specific, passed via `ModelConfig.endpoint`)
2. `ANTHROPIC_ENDPOINT` (provider-specific fallback)
3. Default: `None` (pydantic-ai default: `https://api.anthropic.com`)

For Anthropic API key resolution (existing):
1. `ORCHESTRATOR_API_KEY` / `CYPHER_API_KEY` (role-specific)
2. `ANTHROPIC_API_KEY` (provider-specific)
3. `None` (validation error)

### Edge Cases

#### Empty/Whitespace API Key

The `_resolve_api_key` function now treats empty strings and whitespace-only strings as "not set":

```python
# These should all fall through to env var or return None:
api_key = None
api_key = ""
api_key = "   "
api_key = "\t\n"
```

This prevents empty `Bearer ` headers from being sent to the API.

#### Proxy Considerations

When a custom endpoint is configured:
- The proxy settings should NOT intercept requests to the custom endpoint
- Users may need to add the custom endpoint domain to `no_proxy`

## Implementation Details

### File Changes

| File | Change |
|------|--------|
| `providers/base.py` | Add `endpoint` slot and parameter to `AnthropicProvider`; update `_resolve_api_key` to handle whitespace |
| `constants.py` | Add `ENV_ANTHROPIC_ENDPOINT` constant |

### Backward Compatibility

- Existing configurations without `ORCHESTRATOR_ENDPOINT` will continue to use `api.anthropic.com`
- The `endpoint` parameter is optional (`str | None`)
- No changes to `ModelConfig` dataclass needed (already has `endpoint` field)

### Testing Strategy

#### Unit Tests

1. Test `AnthropicProvider` with custom endpoint
2. Test `_resolve_api_key` with empty/whitespace strings
3. Test endpoint fallback chain
4. Test API key validation

#### Integration Tests

1. Test full flow with mock Anthropic-compatible endpoint
2. Test error messages for missing API key
3. Test proxy bypass with custom endpoint

#### Manual Testing

1. Configure `ORCHESTRATOR_PROVIDER=anthropic` with custom endpoint
2. Verify requests go to custom endpoint, not `api.anthropic.com`
3. Test with empty `ANTHROPIC_AUTH_TOKEN` (should warn, not crash)

## Error Messages

### User-Facing Guidance

When authentication errors occur, suggest:
1. Verify `ORCHESTRATOR_API_KEY` is properly configured (not empty or whitespace)
2. If using custom endpoint, ensure `ORCHESTRATOR_ENDPOINT` is set
3. Add custom endpoint domain to `no_proxy` if using proxy

## Migration Path

### For Users with Custom Anthropic Endpoints

No changes needed - the existing `.env` configuration will now work:

```env
ORCHESTRATOR_PROVIDER=anthropic
ORCHESTRATOR_MODEL=GLM-5
ORCHESTRATOR_API_KEY=your-key
ORCHESTRATOR_ENDPOINT=https://your-custom-endpoint.com/anthropic
```

### For Users Behind Proxies

Add custom endpoint to `no_proxy`:

```bash
export no_proxy="localhost,127.0.0.1,*.your-custom-endpoint.com"
```

## Success Criteria

1. Users can configure custom Anthropic-compatible endpoints via `ORCHESTRATOR_ENDPOINT`
2. Empty/whitespace API keys are rejected early (affects all providers)
3. All existing tests pass
4. New tests cover the edge cases
5. Error messages are clear and actionable

## Related Issues

- Dynamic model catalog integration (`.specs/dynamic_env_models_integration_spec.md`)
- Provider endpoint support (existing for OpenAI, Google, Azure)
