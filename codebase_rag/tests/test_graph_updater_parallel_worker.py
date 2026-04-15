from pathlib import Path

from codebase_rag.graph_updater import GraphUpdater


def test_process_worker_chunk_emits_file_nodes_for_non_code_files(tmp_path: Path) -> None:
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