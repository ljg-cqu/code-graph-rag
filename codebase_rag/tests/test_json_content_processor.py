from pathlib import Path

from codebase_rag import constants as cs
from codebase_rag.parsers.json_content_processor import JsonContentProcessor
from codebase_rag.services.memory_ingestor import MemoryIngestor


def _get_nodes_by_label(
    ingestor: MemoryIngestor,
    label: str,
) -> list[dict[str, object]]:
    return [props for node_label, props in ingestor.nodes if node_label == label]


def _get_relationships(
    ingestor: MemoryIngestor,
    rel_type: str,
) -> list[dict[str, object]]:
    relationships: list[dict[str, object]] = []
    for (
        from_label,
        from_key,
        current_rel_type,
        to_label,
        to_key,
    ), rows in ingestor.relationships.items():
        if current_rel_type != rel_type:
            continue
        for row in rows:
            relationships.append(
                {
                    "from_label": from_label,
                    "from_key": from_key,
                    "to_label": to_label,
                    "to_key": to_key,
                    "from_val": row["from_val"],
                    "to_val": row["to_val"],
                    "props": row["props"],
                }
            )
    return relationships


def test_process_json_file_models_object_field_values(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    config_path = repo_path / "config"
    config_path.mkdir()
    json_path = config_path / "settings.json"
    json_path.write_text(
        '{"service.config": {"port": 8080}, "enabled": true}',
        encoding=cs.ENCODING_UTF8,
    )

    ingestor = MemoryIngestor()
    processor = JsonContentProcessor(ingestor, repo_path, "repo")

    processor.process_json_file(json_path)

    field_nodes = _get_nodes_by_label(ingestor, cs.NodeLabel.JSON_FIELD)
    field_by_key = {str(props[cs.KEY_JSON_KEY]): props for props in field_nodes}
    service_field = field_by_key["service.config"]
    enabled_field = field_by_key["enabled"]
    has_value_rels = _get_relationships(ingestor, cs.RelationshipType.HAS_VALUE)

    assert "service.config" not in str(service_field[cs.KEY_QUALIFIED_NAME])
    assert enabled_field[cs.KEY_JSON_VALUE] == "true"
    assert enabled_field[cs.KEY_JSON_VALUE_TYPE] == "boolean"
    assert any(
        rel["from_val"] == service_field[cs.KEY_QUALIFIED_NAME]
        and rel["to_label"] == cs.NodeLabel.JSON_OBJECT
        for rel in has_value_rels
    )
    assert any(
        rel["from_val"] == enabled_field[cs.KEY_QUALIFIED_NAME]
        and rel["to_label"] == cs.NodeLabel.JSON_VALUE
        for rel in has_value_rels
    )


def test_process_json_file_models_root_array_with_indexed_elements(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    json_path = repo_path / "array.json"
    json_path.write_text('[1, {"name": "x"}, [false]]', encoding=cs.ENCODING_UTF8)

    ingestor = MemoryIngestor()
    processor = JsonContentProcessor(ingestor, repo_path, "repo")

    processor.process_json_file(json_path)

    contains_json_rels = _get_relationships(ingestor, cs.RelationshipType.CONTAINS_JSON)
    has_element_rels = _get_relationships(ingestor, cs.RelationshipType.HAS_ELEMENT)
    root_rel = contains_json_rels[0]
    root_array_qn = str(root_rel["to_val"])
    root_array_rels = [
        rel for rel in has_element_rels if rel["from_val"] == root_array_qn
    ]

    assert root_rel["to_label"] == cs.NodeLabel.JSON_ARRAY
    assert {rel["props"][cs.KEY_INDEX] for rel in root_array_rels} == {0, 1, 2}
    assert {rel["to_label"] for rel in root_array_rels} == {
        cs.NodeLabel.JSON_VALUE,
        cs.NodeLabel.JSON_OBJECT,
        cs.NodeLabel.JSON_ARRAY,
    }


def test_process_json_file_supports_root_scalars_and_distinct_paths(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    config_path = repo_path / "config"
    deploy_path = repo_path / "deploy"
    config_path.mkdir()
    deploy_path.mkdir()
    first_path = config_path / "server.json"
    second_path = deploy_path / "server.json"
    first_path.write_text("1", encoding=cs.ENCODING_UTF8)
    second_path.write_text("2", encoding=cs.ENCODING_UTF8)

    ingestor = MemoryIngestor()
    processor = JsonContentProcessor(ingestor, repo_path, "repo")

    processor.process_json_file(first_path)
    processor.process_json_file(second_path)

    contains_json_rels = _get_relationships(ingestor, cs.RelationshipType.CONTAINS_JSON)
    root_targets = {str(rel["to_val"]) for rel in contains_json_rels}

    assert {rel["to_label"] for rel in contains_json_rels} == {cs.NodeLabel.JSON_VALUE}
    assert len(root_targets) == 2
