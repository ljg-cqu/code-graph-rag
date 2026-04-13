"""Unit tests for JSON Data Ingestion feature."""

import json
import tempfile
from pathlib import Path

import pytest

from codebase_rag.json_ingestion import load_json_files, validate_json_input

SAMPLE_VALID_JSON = {
    "metadata": {"dataset_id": "test_dataset_1", "source": "test_source"},
    "entities": [
        {
            "id": "ent_001",
            "labels": ["TestEntity", "Person"],
            "properties": {
                "name": "John Doe",
                "description": "Test person entity",
                "age": 30,
            },
        },
        {
            "id": "ent_002",
            "labels": ["TestEntity", "Company"],
            "properties": {"name": "Acme Corp", "description": "Test company entity"},
        },
    ],
    "relationships": [
        {
            "source_entity_id": "ent_001",
            "target_entity_id": "ent_002",
            "type": "WORKS_AT",
            "properties": {"description": "John works at Acme Corp", "since": 2020},
        }
    ],
}


def test_validate_valid_json():
    """Test validation of properly structured JSON input."""
    valid, validated, errors = validate_json_input(SAMPLE_VALID_JSON)
    assert valid is True
    assert len(errors) == 0
    assert validated is not None
    assert validated.metadata.dataset_id == "test_dataset_1"
    assert len(validated.entities) == 2
    assert len(validated.relationships) == 1


def test_validate_json_missing_required_fields():
    """Test validation fails when required fields are missing."""
    # Missing entity id
    invalid_json = SAMPLE_VALID_JSON.copy()
    invalid_json["entities"][0].pop("id")
    valid, validated, errors = validate_json_input(invalid_json)
    assert valid is False
    assert len(errors) > 0

    # Missing entity name in properties
    invalid_json = SAMPLE_VALID_JSON.copy()
    invalid_json["entities"][0]["properties"].pop("name")
    valid, validated, errors = validate_json_input(invalid_json)
    assert valid is False
    assert len(errors) > 0

    # Missing entity description in properties
    invalid_json = SAMPLE_VALID_JSON.copy()
    invalid_json["entities"][0]["properties"].pop("description")
    valid, validated, errors = validate_json_input(invalid_json)
    assert valid is False
    assert len(errors) > 0

    # Missing relationship source_entity_id
    invalid_json = SAMPLE_VALID_JSON.copy()
    invalid_json["relationships"][0].pop("source_entity_id")
    valid, validated, errors = validate_json_input(invalid_json)
    assert valid is False
    assert len(errors) > 0


def test_validate_duplicate_entity_ids():
    """Test validation fails when duplicate entity IDs are present."""
    invalid_json = SAMPLE_VALID_JSON.copy()
    invalid_json["entities"][1]["id"] = "ent_001"  # Duplicate ID
    valid, validated, errors = validate_json_input(invalid_json)
    assert valid is False
    assert "duplicate" in errors[0].lower()


def test_validate_invalid_relationship_references():
    """Test validation fails when relationships reference non-existent entities."""
    invalid_json = SAMPLE_VALID_JSON.copy()
    invalid_json["relationships"][0]["source_entity_id"] = "non_existent_id"
    valid, validated, errors = validate_json_input(invalid_json)
    assert valid is False
    assert "non-existent" in errors[0].lower()


def test_load_json_files_single_file():
    """Test loading JSON from a single file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_VALID_JSON, f)
        temp_path = Path(f.name)

    try:
        files = load_json_files(str(temp_path))
        assert len(files) == 1
        assert files[0][0] == temp_path
        assert files[0][1]["metadata"]["dataset_id"] == "test_dataset_1"
    finally:
        temp_path.unlink()


def test_load_json_files_directory():
    """Test loading JSON files from a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create 2 JSON files
        for i in range(2):
            file_path = tmp_path / f"test_{i}.json"
            data = SAMPLE_VALID_JSON.copy()
            data["metadata"]["dataset_id"] = f"test_{i}"
            with open(file_path, "w") as f:
                json.dump(data, f)

        # Create a non-JSON file that should be ignored
        (tmp_path / "not_json.txt").write_text("ignore me")

        files = load_json_files(str(tmp_path))
        assert len(files) == 2
        dataset_ids = {f[1]["metadata"]["dataset_id"] for f in files}
        assert dataset_ids == {"test_0", "test_1"}


def test_load_json_files_invalid_path():
    """Test error is raised for invalid input path."""
    with pytest.raises(ValueError):
        load_json_files("/non/existent/path/12345.json")

    with pytest.raises(ValueError):
        load_json_files("/non/existent/directory/")


def test_handle_json_update_event_basic():
    """Test basic event handling functionality."""
    from codebase_rag.json_ingestion import handle_json_update_event

    event = {
        "id": "evt_123",
        "operation": "add",
        "timestamp": "2024-01-01T00:00:00Z",
        "entities": SAMPLE_VALID_JSON["entities"],
        "relationships": SAMPLE_VALID_JSON["relationships"],
    }

    # Dry run to avoid database writes
    result = handle_json_update_event(event, "test_dataset", dry_run=True)
    assert result.event_id == "evt_123"
    assert result.operation == "add"
    assert result.dataset_id == "test_dataset"
    assert result.dry_run is True
