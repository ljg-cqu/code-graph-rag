from pathlib import Path

from codebase_rag import constants as cs
from codebase_rag.graph_updater import GraphUpdater


def test_process_worker_chunk_emits_file_nodes_for_non_code_files(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    readme_path = repo_path / "README.md"
    readme_path.write_text("# hello\n")

    nodes, relationships = GraphUpdater._process_worker_chunk(
        [readme_path],
        repo_path,
        {},
        "repo",
    )

    assert any(
        node["label"] == "File" and node["props"].get("path") == "README.md"
        for node in nodes
    )
    assert any(
        rel["rel_type"] == "CONTAINS_FILE" and rel["to_val"] == "README.md"
        for rel in relationships
    )


def test_process_worker_chunk_uses_package_parent_for_files_in_packages(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    package_path = repo_path / "pkg"
    package_path.mkdir()
    file_path = package_path / "config.json"
    file_path.write_text("{}")

    nodes, relationships = GraphUpdater._process_worker_chunk(
        [file_path],
        repo_path,
        {Path("pkg"): "repo.pkg"},
        "repo",
    )

    assert any(
        node["label"] == "File" and node["props"].get("path") == "pkg/config.json"
        for node in nodes
    )
    assert any(
        rel["rel_type"] == "CONTAINS_FILE"
        and rel["from_label"] == "Package"
        and rel["from_val"] == "repo.pkg"
        and rel["to_val"] == "pkg/config.json"
        for rel in relationships
    )


def test_process_worker_chunk_skips_canonical_json_ingestion_payloads(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    file_path = repo_path / "dataset.json"
    file_path.write_text(
        '{"metadata": {"dataset_id": "demo"}, "entities": [{"name": "Entity"}]}',
        encoding=cs.ENCODING_UTF8,
    )

    nodes, relationships = GraphUpdater._process_worker_chunk(
        [file_path],
        repo_path,
        {},
        "repo",
    )

    assert not any(
        node["label"] in {"JsonObject", "JsonArray", "JsonField", "JsonValue"}
        for node in nodes
    )
    assert not any(rel["rel_type"] == "CONTAINS_JSON" for rel in relationships)


def test_process_worker_chunk_indexes_noncanonical_json_with_entities_keys(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    file_path = repo_path / "settings.json"
    file_path.write_text(
        '{"entities": {"enabled": true}, "relationships": ["a", "b"]}',
        encoding=cs.ENCODING_UTF8,
    )

    nodes, relationships = GraphUpdater._process_worker_chunk(
        [file_path],
        repo_path,
        {},
        "repo",
    )

    assert any(node["label"] == "JsonObject" for node in nodes)
    assert any(rel["rel_type"] == "CONTAINS_JSON" for rel in relationships)
