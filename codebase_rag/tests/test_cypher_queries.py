from codebase_rag.cypher_queries import CYPHER_DELETE_PROJECT


def test_delete_project_query_scopes_json_matches_to_project_files() -> None:
    assert "OPTIONAL MATCH (container)-[:CONTAINS_JSON]->(json_root)" in CYPHER_DELETE_PROJECT
    assert "WHERE container:File" in CYPHER_DELETE_PROJECT


def test_delete_project_query_traverses_has_value_edges() -> None:
    assert "HAS_FIELD|HAS_VALUE|HAS_ELEMENT" in CYPHER_DELETE_PROJECT
