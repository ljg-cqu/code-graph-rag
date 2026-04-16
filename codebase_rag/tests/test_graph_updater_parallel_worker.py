from pathlib import Path

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
