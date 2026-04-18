"""Test the CYPHER_PROJECT_STRUCTURE query specifically."""
from __future__ import annotations

import pytest

from codebase_rag.cypher_queries import CYPHER_PROJECT_STRUCTURE


def test_cypher_project_structure_syntax() -> None:
    """Test that CYPHER_PROJECT_STRUCTURE query has correct syntax without ORDER BY clause."""
    # Verify the query exists and is not empty
    assert CYPHER_PROJECT_STRUCTURE.strip(), "CYPHER_PROJECT_STRUCTURE should not be empty"
    
    # Verify it contains the expected MATCH patterns
    assert "MATCH (p:Project {name: $project_name})" in CYPHER_PROJECT_STRUCTURE
    assert "OPTIONAL MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER*]->(d)" in CYPHER_PROJECT_STRUCTURE
    assert "OPTIONAL MATCH (d)-[:CONTAINS_FILE]->(f:File)" in CYPHER_PROJECT_STRUCTURE
    
    # Verify it does NOT contain the problematic ORDER BY clause (as per spec fix)
    assert "ORDER BY" not in CYPHER_PROJECT_STRUCTURE, "Query should not contain ORDER BY to avoid Memgraph parser bug"
    
    # Verify it returns the expected fields
    assert "d.name AS dir_name" in CYPHER_PROJECT_STRUCTURE
    assert "d.path AS dir_path" in CYPHER_PROJECT_STRUCTURE
    assert "count(DISTINCT f) AS file_count" in CYPHER_PROJECT_STRUCTURE
    assert "count(DISTINCT func) AS function_count" in CYPHER_PROJECT_STRUCTURE
    assert "count(DISTINCT cls) AS class_count" in CYPHER_PROJECT_STRUCTURE


@pytest.mark.integration
def test_cypher_project_structure_execution(memgraph_ingestor) -> None:
    """Test that CYPHER_PROJECT_STRUCTURE query executes without errors."""
    # Create a minimal test project structure
    memgraph_ingestor._execute_query(
        """
        CREATE (p:Project {name: 'test_project'})
        CREATE (pkg:Package {name: 'mypackage', path: 'mypackage'})
        CREATE (f:File {name: 'module.py', path: 'mypackage/module.py', extension: '.py'})
        CREATE (m:Module {qualified_name: 'mypackage.module', path: 'mypackage/module.py'})
        CREATE (func:Function {qualified_name: 'mypackage.module.myfunc', name: 'myfunc'})
        CREATE (cls:Class {qualified_name: 'mypackage.module.MyClass', name: 'MyClass'})
        CREATE (p)-[:CONTAINS_PACKAGE]->(pkg)
        CREATE (pkg)-[:CONTAINS_FILE]->(f)
        CREATE (f)-[:CONTAINS_MODULE]->(m)
        CREATE (m)-[:DEFINES]->(func)
        CREATE (m)-[:DEFINES]->(cls)
        """
    )
    
    # Execute the query
    results = memgraph_ingestor._execute_query(
        CYPHER_PROJECT_STRUCTURE,
        {"project_name": "test_project"}
    )
    
    # Should return results without "Unbound variable" error
    assert len(results) > 0, "Query should return results"
    
    # Check result structure
    for row in results:
        assert "dir_name" in row
        assert "dir_path" in row  
        assert "file_count" in row
        assert "function_count" in row
        assert "class_count" in row
    
    # Clean up
    memgraph_ingestor._execute_query("MATCH (n) DETACH DELETE n")