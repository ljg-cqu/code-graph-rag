"""Unit tests for canonical JSON ingestion behavior."""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from codebase_rag.config import settings
from codebase_rag.json_ingestion import (
    handle_json_update_event,
    ingest_json_data,
    load_json_files,
    recreate_json_vector_index,
    validate_json_input,
)

SAMPLE_VALID_JSON = {
    "metadata": {"dataset_id": "test_dataset_1", "source": "test_source"},
    "entities": [
        {
            "id": "ent_001",
            "name": "John Doe",
            "type": "Person",
            "labels": ["TestEntity", "Person"],
            "properties": {
                "description": "Test person entity",
                "age": 30,
            },
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


def test_validate_valid_json() -> None:
    valid, validated, errors = validate_json_input(deepcopy(SAMPLE_VALID_JSON))

    assert valid is True
    assert errors == []
    assert validated is not None
    assert validated["metadata"]["dataset_id"] == "test_dataset_1"
    assert len(validated["entities"]) == 2
    assert len(validated["relationships"]) == 1


def test_validate_json_missing_required_fields() -> None:
    invalid_json = deepcopy(SAMPLE_VALID_JSON)
    invalid_json["entities"][0].pop("name")
    valid, _, errors = validate_json_input(invalid_json)
    assert valid is False
    assert errors

    invalid_json = deepcopy(SAMPLE_VALID_JSON)
    invalid_json["relationships"][0].pop("source")
    valid, _, errors = validate_json_input(invalid_json)
    assert valid is False
    assert errors

    invalid_json = deepcopy(SAMPLE_VALID_JSON)
    invalid_json["relationships"][0].pop("relationship")
    valid, _, errors = validate_json_input(invalid_json)
    assert valid is False
    assert errors


def test_validate_auto_generates_missing_entity_ids() -> None:
    input_json = deepcopy(SAMPLE_VALID_JSON)
    input_json["entities"][0].pop("id")
    input_json["relationships"][0]["source"] = "John Doe"

    valid, validated, errors = validate_json_input(input_json)

    assert valid is True
    assert errors == []
    assert validated is not None
    assert validated["entities"][0]["id"] == "john_doe"


def test_validate_duplicate_entity_ids() -> None:
    invalid_json = deepcopy(SAMPLE_VALID_JSON)
    invalid_json["entities"][1]["id"] = "ent_001"

    valid, _, errors = validate_json_input(invalid_json)

    assert valid is False
    assert any("duplicate" in error.lower() for error in errors)


def test_validate_invalid_relationship_references() -> None:
    invalid_json = deepcopy(SAMPLE_VALID_JSON)
    invalid_json["relationships"][0]["source"] = "non_existent_id"

    valid, _, errors = validate_json_input(invalid_json)

    assert valid is False
    assert any("non-existent" in error.lower() for error in errors)


def test_validate_relationship_wrapper_object() -> None:
    input_json = deepcopy(SAMPLE_VALID_JSON)
    input_json["relationships"] = {"relationships": input_json["relationships"]}

    valid, validated, errors = validate_json_input(input_json)

    assert valid is True
    assert errors == []
    assert validated is not None
    assert len(validated["relationships"]) == 1
    assert validated["relationships"][0]["relationship"] == "WORKS_AT"


def test_load_json_files_single_file() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as json_file:
        json.dump(SAMPLE_VALID_JSON, json_file)
        temp_path = Path(json_file.name)

    try:
        files = load_json_files(str(temp_path))
        assert len(files) == 1
        assert files[0][0] == temp_path
        assert files[0][1]["metadata"]["dataset_id"] == "test_dataset_1"
    finally:
        temp_path.unlink()


def test_load_json_files_directory() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for index in range(2):
            file_path = temp_path / f"test_{index}.json"
            data = deepcopy(SAMPLE_VALID_JSON)
            data["metadata"]["dataset_id"] = f"test_{index}"
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file)

        (temp_path / "not_json.txt").write_text("ignore me", encoding="utf-8")

        files = load_json_files(str(temp_path))
        assert len(files) == 2
        dataset_ids = {file_data[1]["metadata"]["dataset_id"] for file_data in files}
        assert dataset_ids == {"test_0", "test_1"}


def test_load_json_files_directory_skips_artifacts_and_non_ingestion_json() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        valid_path = temp_path / "valid.json"
        with open(valid_path, "w", encoding="utf-8") as json_file:
            json.dump(SAMPLE_VALID_JSON, json_file)

        cache_dir = temp_path / ".embedding_cache"
        cache_dir.mkdir()
        with open(
            cache_dir / ".tmp_cache_123.json", "w", encoding="utf-8"
        ) as json_file:
            json.dump(SAMPLE_VALID_JSON, json_file)

        egg_info_dir = temp_path / "demo.egg-info"
        egg_info_dir.mkdir()
        with open(egg_info_dir / "metadata.json", "w", encoding="utf-8") as json_file:
            json.dump(SAMPLE_VALID_JSON, json_file)

        optimize_dir = temp_path / "optimize"
        optimize_dir.mkdir()
        with open(
            optimize_dir / "memory_profile_results.json", "w", encoding="utf-8"
        ) as json_file:
            json.dump({"metadata": {"workspace": "default"}}, json_file)

        files = load_json_files(str(temp_path))

        assert len(files) == 1
        assert files[0][0] == valid_path


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


def test_ingest_json_data_dry_run_reports_invalid_json_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        valid_path = temp_path / "valid.json"
        with open(valid_path, "w", encoding="utf-8") as json_file:
            json.dump(SAMPLE_VALID_JSON, json_file)

        invalid_path = temp_path / "invalid.json"
        invalid_path.write_text('{"entities": [}', encoding="utf-8")

        result = ingest_json_data(
            input_path=str(temp_path),
            dry_run=True,
            parallel_workers=1,
        )

        assert result.files_processed == 1
        assert result.files_skipped == 1
        assert any(str(invalid_path) in error for error in result.errors)


def test_load_json_files_invalid_path() -> None:
    with pytest.raises(ValueError):
        load_json_files("/non/existent/path/12345.json")

    with pytest.raises(ValueError):
        load_json_files("/non/existent/directory/")


def test_ingest_json_data_dry_run_with_sample_json() -> None:
    sample_path = Path(__file__).resolve().parents[2] / "sample_json_ingest.json"

    result = ingest_json_data(
        input_path=str(sample_path),
        dry_run=True,
        parallel_workers=1,
    )

    assert result.files_processed == 1
    assert result.files_skipped == 0
    assert result.entities_processed == 3
    assert result.relationships_processed == 2
    assert result.entities_ingested == 3
    assert result.relationships_ingested == 2
    assert result.errors == []


def test_ingest_json_data_dry_run_resolves_cross_file_relationships() -> None:
    job_file = {
        "metadata": {"dataset_id": "shared_dataset"},
        "entities": [
            {
                "id": "job_001",
                "name": "Backend Engineer",
                "type": "JobPosting",
                "properties": {"description": "A backend engineering role."},
            }
        ],
        "relationships": [
            {
                "source": "job_001",
                "target": "skill_001",
                "relationship": "REQUIRES_SKILL",
            }
        ],
    }
    skill_file = {
        "metadata": {"dataset_id": "shared_dataset"},
        "entities": [
            {
                "id": "skill_001",
                "name": "Python",
                "type": "Skill",
                "properties": {"description": "Python programming language."},
            }
        ],
        "relationships": [],
    }

    result = ingest_json_data(
        pre_loaded_data=[
            (Path("job.json"), job_file),
            (Path("skill.json"), skill_file),
        ],
        dry_run=True,
        parallel_workers=2,
    )

    assert result.files_processed == 2
    assert result.entities_processed == 2
    assert result.relationships_processed == 1
    assert result.entities_ingested == 2
    assert result.relationships_ingested == 1
    assert result.errors == []


def test_ingest_json_data_dry_run_rejects_missing_prepared_references() -> None:
    job_file = {
        "metadata": {"dataset_id": "shared_dataset"},
        "entities": [
            {
                "id": "job_001",
                "name": "Backend Engineer",
                "type": "JobPosting",
                "properties": {"description": "A backend engineering role."},
            }
        ],
        "relationships": [
            {
                "source": "job_001",
                "target": "skill_999",
                "relationship": "REQUIRES_SKILL",
            }
        ],
    }

    result = ingest_json_data(
        pre_loaded_data=[(Path("job.json"), job_file)],
        dry_run=True,
        parallel_workers=1,
    )

    assert result.files_processed == 1
    assert result.entities_processed == 1
    assert result.relationships_processed == 1
    assert result.entities_ingested == 0
    assert result.relationships_ingested == 0
    assert result.errors == [
        "job.json: Relationship 1 target references non-existent entity in dataset 'shared_dataset': skill_999"
    ]


def test_ingest_json_data_dry_run_rejects_ambiguous_prepared_names() -> None:
    job_file = {
        "metadata": {"dataset_id": "shared_dataset"},
        "entities": [
            {
                "id": "job_001",
                "name": "Backend Engineer",
                "type": "JobPosting",
                "properties": {"description": "A backend engineering role."},
            }
        ],
        "relationships": [
            {
                "source": "job_001",
                "target": "Python",
                "relationship": "REQUIRES_SKILL",
            }
        ],
    }
    skill_file_a = {
        "metadata": {"dataset_id": "shared_dataset"},
        "entities": [
            {
                "id": "skill_001",
                "name": "Python",
                "type": "Skill",
                "properties": {"description": "Python programming language."},
            }
        ],
        "relationships": [],
    }
    skill_file_b = {
        "metadata": {"dataset_id": "shared_dataset"},
        "entities": [
            {
                "id": "skill_002",
                "name": "Python",
                "type": "Skill",
                "properties": {"description": "Python web framework skill."},
            }
        ],
        "relationships": [],
    }

    result = ingest_json_data(
        pre_loaded_data=[
            (Path("job.json"), job_file),
            (Path("skill_a.json"), skill_file_a),
            (Path("skill_b.json"), skill_file_b),
        ],
        dry_run=True,
        parallel_workers=3,
    )

    assert result.files_processed == 3
    assert result.entities_processed == 3
    assert result.relationships_processed == 1
    assert result.entities_ingested == 0
    assert result.relationships_ingested == 0
    assert result.errors == [
        "job.json: Relationship 1 target references ambiguous entity name in dataset 'shared_dataset': Python"
    ]


def test_handle_json_update_event_basic() -> None:
    event = {
        "id": "evt_123",
        "operation": "add",
        "timestamp": "2024-01-01T00:00:00Z",
        "entities": SAMPLE_VALID_JSON["entities"],
        "relationships": SAMPLE_VALID_JSON["relationships"],
    }

    result = handle_json_update_event(event, "test_dataset", dry_run=True)

    assert result.event_id == "evt_123"
    assert result.operation == "add"
    assert result.dataset_id == "test_dataset"
    assert result.dry_run is True
    assert result.errors == []


def test_recreate_json_vector_index_recreates_mismatched_dimension() -> None:
    class FakeIngestor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def __enter__(self) -> FakeIngestor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def fetch_all(self, query: str, params: dict | None = None) -> list[dict]:
            self.calls.append(("fetch_all", " ".join(query.split())))
            return [
                {
                    "index_name": settings.JSON_MEMGRAPH_VECTOR_INDEX_NAME,
                    "dimension": 768,
                }
            ]

        def execute_write(self, query: str, params: dict | None = None) -> None:
            self.calls.append(("execute_write", " ".join(query.split())))

    fake_ingestor = FakeIngestor()

    with patch(
        "codebase_rag.json_ingestion._create_json_ingestor",
        return_value=fake_ingestor,
    ):
        recreate_json_vector_index(batch_size=10, dimension=1024)

    write_queries = [query for kind, query in fake_ingestor.calls if kind == "execute_write"]
    assert any("MATCH (n:JsonEntity) SET n.embedding = NULL" in query for query in write_queries)
    assert any(
        f"DROP VECTOR INDEX {settings.JSON_MEMGRAPH_VECTOR_INDEX_NAME};" in query
        for query in write_queries
    )
    assert any(
        f"CREATE VECTOR INDEX {settings.JSON_MEMGRAPH_VECTOR_INDEX_NAME}" in query
        and '"dimension": 1024' in query
        for query in write_queries
    )


def test_ingest_json_data_calls_ensure_constraints() -> None:
    class FakeIngestor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __enter__(self) -> FakeIngestor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def ensure_constraints(self) -> None:
            self.calls.append("ensure_constraints")

        def fetch_all(self, query: str, params: dict | None = None) -> list[dict]:
            return []

        def execute_write(self, query: str, params: dict | None = None) -> None:
            pass

    fake_ingestor = FakeIngestor()

    with patch(
        "codebase_rag.json_ingestion._create_json_ingestor",
        return_value=fake_ingestor,
    ):
        result = ingest_json_data(pre_loaded_data=[(Path("test.json"), deepcopy(SAMPLE_VALID_JSON))])

    assert "ensure_constraints" in fake_ingestor.calls


def test_recreate_json_vector_index_keeps_matching_dimension() -> None:
    class FakeIngestor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def __enter__(self) -> FakeIngestor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def fetch_all(self, query: str, params: dict | None = None) -> list[dict]:
            self.calls.append(("fetch_all", " ".join(query.split())))
            return [
                {
                    "index_name": settings.JSON_MEMGRAPH_VECTOR_INDEX_NAME,
                    "dimension": 1024,
                }
            ]

        def execute_write(self, query: str, params: dict | None = None) -> None:
            self.calls.append(("execute_write", " ".join(query.split())))

    fake_ingestor = FakeIngestor()

    with patch(
        "codebase_rag.json_ingestion._create_json_ingestor",
        return_value=fake_ingestor,
    ):
        recreate_json_vector_index(batch_size=10, dimension=1024)

    write_queries = [query for kind, query in fake_ingestor.calls if kind == "execute_write"]
    assert write_queries == []
