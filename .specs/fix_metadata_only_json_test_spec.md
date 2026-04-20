# Design Spec: Fix Metadata-Only JSON Test Expectation

## Problem Statement

Test `test_load_json_files_single_non_ingestion_file_is_loaded_for_validation` fails because it expects metadata-only JSON files to be loaded by `load_json_files()`, but the JSON purpose detection feature correctly skips them.

```
FAILED codebase_rag/tests/test_json_ingestion.py::test_load_json_files_single_non_ingestion_file_is_loaded_for_validation
assert len(files) == 1
E   assert 0 == 1
```

## Root Cause Analysis

### Timeline

1. **Original behavior**: `load_json_files()` loaded any valid JSON file
2. **Commit `015f2f7`**: Added `_detect_json_purpose()` to classify JSON files and skip non-ingestion payloads
3. **Test not updated**: Test still expects metadata-only files to be loaded

### Why the New Behavior is Correct

1. **Schema compliance**: `ingestion_schema.json` requires `entities` field (line 6: `"required": ["entities"]`)
2. **Metadata-only is invalid**: A file with just `{"metadata": {"workspace": "default"}}` has no entities to ingest
3. **Purpose detection provides guidance**: The `_detect_json_purpose()` function returns helpful guidance for skipped files

### The `_detect_json_purpose` Logic

```python
# Check for metadata-only
if "metadata" in data and "entities" not in data and "relationships" not in data:
    return "metadata_only", "This file contains only metadata. For ingestion, add an 'entities' array with the actual entity definitions."
```

This is correct - metadata-only files should be skipped with guidance.

## Proposed Solution

Update the test to verify that metadata-only files are properly skipped with guidance, not loaded.

### Test Update

**File:** `codebase_rag/tests/test_json_ingestion.py`

Replace:
```python
def test_load_json_files_single_non_ingestion_file_is_loaded_for_validation() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as json_file:
        json.dump({"metadata": {"workspace": "default"}}, json_file)
        temp_path = Path(json_file.name)

    try:
        files = load_json_files(str(temp_path))
        assert len(files) == 1
        assert files[0][0] == temp_path
        assert files[0][1] == {"metadata": {"workspace": "default"}}
    finally:
        temp_path.unlink()
```

With:
```python
def test_load_json_files_metadata_only_file_is_skipped_with_guidance() -> None:
    """Verify metadata-only JSON files are skipped with helpful guidance."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as json_file:
        json.dump({"metadata": {"workspace": "default"}}, json_file)
        temp_path = Path(json_file.name)

    try:
        files = load_json_files(str(temp_path))
        # Metadata-only files should be skipped, not loaded
        assert len(files) == 0
    finally:
        temp_path.unlink()


def test_load_json_files_schema_definition_is_skipped() -> None:
    """Verify JSON Schema files are skipped with guidance."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as json_file:
        json.dump({"$schema": "http://json-schema.org/draft-07/schema#", "properties": {}}, json_file)
        temp_path = Path(json_file.name)

    try:
        files = load_json_files(str(temp_path))
        # Schema definition files should be skipped
        assert len(files) == 0
    finally:
        temp_path.unlink()


def test_load_json_files_config_file_is_skipped() -> None:
    """Verify configuration-like JSON files are skipped with guidance."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as json_file:
        json.dump({"config": {"setting": "value"}}, json_file)
        temp_path = Path(json_file.name)

    try:
        files = load_json_files(str(temp_path))
        # Configuration files should be skipped
        assert len(files) == 0
    finally:
        temp_path.unlink()
```

## Implementation Plan

1. **Update test file**: Replace the failing test with new tests that verify the correct behavior
2. **Add comprehensive coverage**: Test all major non-ingestion JSON patterns detected by `_detect_json_purpose()`

## Testing Strategy

Run the updated tests:
```bash
uv run pytest codebase_rag/tests/test_json_ingestion.py -v -k "metadata_only or schema_definition or config_file"
```

## Success Criteria

- [ ] Test `test_load_json_files_metadata_only_file_is_skipped_with_guidance` passes
- [ ] Test `test_load_json_files_schema_definition_is_skipped` passes
- [ ] Test `test_load_json_files_config_file_is_skipped` passes
- [ ] All other JSON ingestion tests continue to pass
- [ ] No changes to production code required (only test update)

## Related Files

- `codebase_rag/tests/test_json_ingestion.py` - Test file to update
- `codebase_rag/json_ingestion.py` - Implementation (no changes needed)
- `ingestion_schema.json` - Schema definition (reference only)
