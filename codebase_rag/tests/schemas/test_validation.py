"""Tests for the schema validation module."""

from __future__ import annotations

from copy import deepcopy

import pytest

from codebase_rag.schemas import (
    ErrorCodes,
    ValidationResult,
    ValidationError,
    canonical_entity_id,
    validate_ingestion_payload,
    validate_json_input_legacy,
)


SAMPLE_VALID_PAYLOAD = {
    "metadata": {"dataset_id": "test_dataset_1", "source": "test_source"},
    "entities": [
        {
            "id": "ent_001",
            "name": "John Doe",
            "type": "Person",
            "labels": ["TestEntity", "Person"],
            "properties": {"description": "Test person entity", "age": 30},
        },
        {
            "id": "ent_002",
            "name": "Acme Corp",
            "type": "Company",
            "labels": ["TestEntity", "Company"],
            "properties": {"description": "Test company entity"},
        },
    ],
    "relationships": [
        {
            "source": "ent_001",
            "target": "ent_002",
            "relationship": "WORKS_AT",
            "properties": {"description": "John works at Acme Corp", "since": 2020},
        }
    ],
}


class TestCanonicalEntityId:
    """Tests for the canonical_entity_id function."""

    def test_basic_name(self) -> None:
        assert canonical_entity_id("User Service") == "user_service"

    def test_name_with_special_chars(self) -> None:
        assert canonical_entity_id("API-Gateway") == "api_gateway"
        assert canonical_entity_id("User@Service!") == "user_service"

    def test_name_with_multiple_spaces(self) -> None:
        assert canonical_entity_id("User   Service") == "user_service"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            canonical_entity_id("")

    def test_whitespace_only_name_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            canonical_entity_id("   ")


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_to_tuple_converts_errors(self) -> None:
        result = ValidationResult(
            is_valid=False,
            data=None,
            errors=[
                ValidationError(
                    path="entities[0].id",
                    code="entity.id.duplicate",
                    message="Duplicate entity ID: test",
                    severity="error",
                )
            ],
            warnings=[],
        )
        valid, data, errors = result.to_tuple()
        assert valid is False
        assert data is None
        assert errors == ["entities[0].id: Duplicate entity ID: test"]

    def test_has_error_code(self) -> None:
        result = ValidationResult(
            is_valid=False,
            data=None,
            errors=[
                ValidationError(
                    path="entities[0].id",
                    code=ErrorCodes.Entity.ID_DUPLICATE,
                    message="Duplicate entity ID: test",
                    severity="error",
                )
            ],
            warnings=[],
        )
        assert result.has_error_code(ErrorCodes.Entity.ID_DUPLICATE) is True
        assert result.has_error_code(ErrorCodes.Entity.ID_MISSING) is False

    def test_errors_for_path(self) -> None:
        result = ValidationResult(
            is_valid=False,
            data=None,
            errors=[
                ValidationError(
                    path="entities[0].id",
                    code="entity.id.duplicate",
                    message="Error 1",
                    severity="error",
                ),
                ValidationError(
                    path="entities[1].name",
                    code="entity.name.empty",
                    message="Error 2",
                    severity="error",
                ),
            ],
            warnings=[],
        )
        entity_errors = result.errors_for_path("entities[0]")
        assert len(entity_errors) == 1
        assert entity_errors[0].message == "Error 1"


class TestValidateIngestionPayload:
    """Tests for the validate_ingestion_payload function."""

    def test_valid_payload(self) -> None:
        result = validate_ingestion_payload(deepcopy(SAMPLE_VALID_PAYLOAD))
        assert result.is_valid is True
        assert result.data is not None
        assert len(result.errors) == 0

    def test_missing_required_entity_name(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["entities"][0].pop("name")
        result = validate_ingestion_payload(payload)
        assert result.is_valid is False
        assert result.has_error_code(ErrorCodes.Schema.VALIDATION_FAILED)

    def test_missing_required_relationship_source(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["relationships"][0].pop("source")
        result = validate_ingestion_payload(payload)
        assert result.is_valid is False

    def test_auto_generates_missing_entity_ids(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["entities"][0].pop("id")
        payload["relationships"][0]["source"] = "John Doe"

        result = validate_ingestion_payload(payload)

        assert result.is_valid is True
        assert len(result.errors) == 0
        # Should have a warning about auto-generated ID
        assert any(e.code == ErrorCodes.Entity.ID_AUTO_GENERATED for e in result.warnings)
        # Should have auto-generated the ID
        assert result.data is not None
        assert result.data["entities"][0]["id"] == "john_doe"

    def test_auto_generated_id_warning_in_non_strict_mode(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        # Remove the ID but keep a reference by name
        payload["entities"][0].pop("id")
        payload["relationships"][0]["source"] = "John Doe"  # Reference by name

        result = validate_ingestion_payload(payload, strict=False)

        assert result.is_valid is True
        assert any(e.code == ErrorCodes.Entity.ID_AUTO_GENERATED for e in result.warnings)

    def test_auto_generated_id_error_in_strict_mode(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["entities"][0].pop("id")

        result = validate_ingestion_payload(payload, strict=True)

        assert result.is_valid is False
        assert any(e.code == ErrorCodes.Entity.ID_MISSING for e in result.errors)

    def test_duplicate_entity_ids(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["entities"][1]["id"] = "ent_001"

        result = validate_ingestion_payload(payload)

        assert result.is_valid is False
        assert result.has_error_code(ErrorCodes.Entity.ID_DUPLICATE)

    def test_duplicate_entity_names(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["entities"][1]["name"] = "John Doe"

        result = validate_ingestion_payload(payload)

        assert result.is_valid is False
        assert result.has_error_code(ErrorCodes.Entity.NAME_DUPLICATE)

    def test_invalid_relationship_source_reference(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["relationships"][0]["source"] = "non_existent_id"

        result = validate_ingestion_payload(payload)

        assert result.is_valid is False
        assert result.has_error_code(ErrorCodes.Relationship.SOURCE_NOT_FOUND)

    def test_invalid_relationship_target_reference(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["relationships"][0]["target"] = "non_existent_id"

        result = validate_ingestion_payload(payload)

        assert result.is_valid is False
        assert result.has_error_code(ErrorCodes.Relationship.TARGET_NOT_FOUND)

    def test_comma_separated_target_warning_in_non_strict_mode(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["entities"].append({
            "id": "ent_003",
            "name": "Third Entity",
            "type": "Type",
        })
        payload["relationships"][0]["target"] = "ent_002, ent_003"

        result = validate_ingestion_payload(payload, strict=False)

        assert result.is_valid is True
        assert any(e.code == ErrorCodes.Relationship.COMMA_IN_TARGET for e in result.warnings)
        # Should have split the relationship
        assert result.data is not None
        assert len(result.data["relationships"]) == 2

    def test_comma_separated_target_error_in_strict_mode(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["entities"].append({
            "id": "ent_003",
            "name": "Third Entity",
            "type": "Type",
        })
        payload["relationships"][0]["target"] = "ent_002, ent_003"

        result = validate_ingestion_payload(payload, strict=True)

        assert result.is_valid is False
        assert result.has_error_code(ErrorCodes.Relationship.COMMA_IN_TARGET)

    def test_relationship_wrapper_object(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["relationships"] = {"relationships": payload["relationships"]}

        result = validate_ingestion_payload(payload)

        assert result.is_valid is True
        assert result.data is not None
        assert len(result.data["relationships"]) == 1

    def test_allow_external_references(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["relationships"][0]["source"] = "external_entity"

        result = validate_ingestion_payload(
            payload, allow_external_references=True
        )

        assert result.is_valid is True


class TestValidateJsonInputLegacy:
    """Tests for the legacy validate_json_input_legacy adapter."""

    def test_returns_tuple_format(self) -> None:
        valid, data, errors = validate_json_input_legacy(deepcopy(SAMPLE_VALID_PAYLOAD))

        assert valid is True
        assert data is not None
        assert errors == []

    def test_appends_to_existing_errors(self) -> None:
        existing_errors: list[str] = ["Previous error"]
        valid, data, errors = validate_json_input_legacy(
            deepcopy(SAMPLE_VALID_PAYLOAD), errors=existing_errors
        )

        assert valid is True
        assert errors == ["Previous error"]

    def test_invalid_payload_returns_errors(self) -> None:
        payload = deepcopy(SAMPLE_VALID_PAYLOAD)
        payload["entities"][1]["id"] = "ent_001"  # Duplicate ID
        # Clear relationships to isolate the duplicate ID error
        payload["relationships"] = []

        valid, data, errors = validate_json_input_legacy(payload)

        assert valid is False
        assert data is None
        assert len(errors) == 1
        assert "duplicate" in errors[0].lower()
