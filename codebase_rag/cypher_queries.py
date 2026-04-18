from .constants import CYPHER_DEFAULT_LIMIT

CYPHER_DELETE_ALL = "MATCH (n) DETACH DELETE n;"

CYPHER_LIST_PROJECTS = "MATCH (p:Project) RETURN p.name AS name ORDER BY p.name"

CYPHER_DELETE_PROJECT = """
MATCH (p:Project {name: $project_name})
OPTIONAL MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER|CONTAINS_FILE|CONTAINS_MODULE*]->(container)
OPTIONAL MATCH (container)-[:DEFINES|DEFINES_METHOD*]->(defined)
OPTIONAL MATCH (container)-[:CONTAINS_JSON]->(json_root)
WHERE container:File
OPTIONAL MATCH (json_root)-[:HAS_FIELD|HAS_VALUE|HAS_ELEMENT*]->(json_content)
DETACH DELETE p, container, defined, json_root, json_content
"""

# ─────────────────────────────────────────────────────────
# EXAMPLE QUERIES FOR DEMONSTRATION
# ─────────────────────────────────────────────────────────

CYPHER_EXAMPLE_DECORATED_FUNCTIONS = f"""MATCH (n:Function|Method)
WHERE ANY(d IN n.decorators WHERE toLower(d) IN ['flow', 'task'])
RETURN n.name AS name, n.qualified_name AS qualified_name, labels(n) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_CONTENT_BY_PATH = f"""MATCH (n)
WHERE n.path IS NOT NULL AND n.path STARTS WITH 'workflows'
RETURN n.name AS name, n.path AS path, labels(n) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_KEYWORD_SEARCH = f"""MATCH (n)
WHERE toLower(n.name) CONTAINS 'database' OR (n.qualified_name IS NOT NULL AND toLower(n.qualified_name) CONTAINS 'database')
RETURN n.name AS name, n.qualified_name AS qualified_name, labels(n) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_FIND_FILE = """MATCH (f:File) WHERE toLower(f.name) = 'readme.md' AND f.path = 'README.md'
RETURN f.path as path, f.name as name, labels(f) as type"""

CYPHER_EXAMPLE_README = f"""MATCH (f:File)
WHERE toLower(f.name) CONTAINS 'readme'
RETURN f.path AS path, f.name AS name, labels(f) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_PYTHON_FILES = f"""MATCH (f:File)
WHERE f.extension = '.py'
RETURN f.path AS path, f.name AS name, labels(f) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_TASKS = f"""MATCH (n:Function|Method)
WHERE 'task' IN n.decorators
RETURN n.qualified_name AS qualified_name, n.name AS name, labels(n) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_FILES_IN_FOLDER = f"""MATCH (f:File)
WHERE f.path STARTS WITH 'services'
RETURN f.path AS path, f.name AS name, labels(f) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_LIMIT_ONE = """MATCH (f:File) RETURN f.path as path, f.name as name, labels(f) as type LIMIT 1"""

CYPHER_EXAMPLE_CLASS_METHODS = f"""MATCH (c:Class)-[:DEFINES_METHOD]->(m:Method)
WHERE c.name = 'UserService'
RETURN c.name AS className, m.name AS methodName, m.qualified_name AS qualified_name, labels(m) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXPORT_NODES = """
MATCH (n)
RETURN id(n) as node_id, labels(n) as labels, properties(n) as properties
"""

CYPHER_EXPORT_RELATIONSHIPS = """
MATCH (a)-[r]->(b)
RETURN id(a) as from_id, id(b) as to_id, type(r) as type, properties(r) as properties
"""

CYPHER_RETURN_COUNT = "RETURN count(r) as created"
CYPHER_SET_PROPS_RETURN_COUNT = "SET r += row.props\nRETURN count(r) as created"

CYPHER_GET_FUNCTION_SOURCE_LOCATION = """
MATCH (m:Module)-[:DEFINES]->(n)
WHERE id(n) = $node_id
RETURN n.qualified_name AS qualified_name, n.start_line AS start_line,
       n.end_line AS end_line, m.path AS path
"""

CYPHER_FIND_BY_QUALIFIED_NAME = """
MATCH (n) WHERE n.qualified_name = $qn
OPTIONAL MATCH (m:Module)-[*]-(n)
RETURN n.name AS name, n.start_line AS start, n.end_line AS end, m.path AS path, n.docstring AS docstring
LIMIT 1
"""

# (H) Graph navigation queries
CYPHER_FIND_CALLERS = """
MATCH (caller:Function|Method)-[:CALLS]->(target)
WHERE target.qualified_name = $qn
OPTIONAL MATCH (m:Module)-[:DEFINES]->(caller)
RETURN caller.qualified_name AS qualified_name, caller.name AS name,
       labels(caller) AS type, m.path AS path, caller.start_line AS start_line
ORDER BY caller.qualified_name
"""

CYPHER_FIND_IMPORTERS = """
MATCH (importer:Module)-[:IMPORTS]->(target:Module)
WHERE target.qualified_name = $qn OR target.path = $qn
RETURN importer.qualified_name AS qualified_name, importer.path AS path
ORDER BY importer.qualified_name
"""

CYPHER_FIND_IMPLEMENTATIONS = """
MATCH (impl:Class)-[r:IMPLEMENTS|INHERITS*1..2]->(base)
WHERE base.qualified_name = $qn OR base.name = $qn
OPTIONAL MATCH (m:Module)-[:DEFINES]->(impl)
RETURN DISTINCT impl.qualified_name AS qualified_name, impl.name AS name,
       type(r[0]) AS relationship_type, m.path AS path, impl.start_line AS start_line
ORDER BY impl.qualified_name
"""

# ─────────────────────────────────────────────────────────
# CYPHER QUERY TEMPLATES FOR FALLBACK GENERATION
# ─────────────────────────────────────────────────────────
# Pre-built parameterized Cypher queries for common patterns.
# Used by QueryMethodOrchestrator when LLM-based generation fails.
#
# NOTE: All templates use WHERE IN clause for label filtering instead of
# `|` union syntax (e.g., `Function|Class|Method`) because the WHERE IN
# approach works correctly even when some labels don't exist in the graph.
# Memgraph simply returns no nodes for non-existent labels.
#
# The three labels that always exist (Function, Class, Method) are listed
# first for optimal query planning.

CYPHER_QUERY_TEMPLATES: dict[str, tuple[str, dict[str, type]]] = {
    "find_by_name": (
        """
        MATCH (n)
        WHERE labels(n)[0] IN ['Function', 'Class', 'Method', 'Enum', 'Type',
                                'Union', 'Interface', 'Contract', 'Library']
          AND (n.name CONTAINS $keyword OR n.qualified_name CONTAINS $keyword)
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type,
               n.path AS file_path, n.start_line AS start_line,
               n.end_line AS end_line
        LIMIT $limit
        """,
        {"keyword": str, "limit": int},
    ),
    "find_by_docstring": (
        """
        MATCH (n)
        WHERE labels(n)[0] IN ['Function', 'Class', 'Method']
          AND n.docstring IS NOT NULL
          AND n.docstring CONTAINS $keyword
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type,
               n.path AS file_path, n.docstring AS docstring
        LIMIT $limit
        """,
        {"keyword": str, "limit": int},
    ),
    "find_dependencies": (
        """
        MATCH (n:Function|Class|Method)-[:CALLS]->(m)
        WHERE n.qualified_name CONTAINS $keyword
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type, n.path AS file_path,
               id(m) AS target_id, m.qualified_name AS target_name
        LIMIT $limit
        """,
        {"keyword": str, "limit": int},
    ),
    # find_callers_of and find_importers_of reference the existing constants
    # which are parameterized queries (require $qn parameter)
    "find_callers_of": (
        CYPHER_FIND_CALLERS,
        {"qn": str},
    ),
    "find_importers_of": (
        CYPHER_FIND_IMPORTERS,
        {"qn": str},
    ),
    # find_by_type uses {label} placeholder interpolated at runtime with NodeLabel validation
    # Cypher doesn't support parameterized node labels, so string interpolation is required
    "find_by_type": (
        """
        MATCH (n:{label})
        WHERE n.project_name = $project
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type, n.path AS file_path
        LIMIT $limit
        """,
        # Note: {label} is interpolated at runtime, NOT via Cypher parameters
        # Validated against NodeLabel enum before interpolation to prevent injection
        {"label": str, "project": str, "limit": int},
    ),
}

CYPHER_PROJECT_STRUCTURE = """
MATCH (p:Project {name: $project_name})
OPTIONAL MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER*]->(d)
OPTIONAL MATCH (d)-[:CONTAINS_FILE]->(f:File)
OPTIONAL MATCH (f)-[:CONTAINS_MODULE]->(m:Module)-[:DEFINES]->(func:Function)
OPTIONAL MATCH (m)-[:DEFINES]->(cls:Class)
RETURN d.name AS dir_name, d.path AS dir_path,
       count(DISTINCT f) AS file_count,
       count(DISTINCT func) AS function_count,
       count(DISTINCT cls) AS class_count
"""


CYPHER_STATS_NODE_COUNTS = """
MATCH (n)
RETURN labels(n) AS labels, count(*) AS count
ORDER BY count DESC
"""

CYPHER_STATS_RELATIONSHIP_COUNTS = """
MATCH ()-[r]->()
RETURN type(r) AS type, count(*) AS count
ORDER BY count DESC
"""


def wrap_with_unwind(query: str) -> str:
    return f"UNWIND $batch AS row\n{query}"


def build_nodes_by_ids_query(node_ids: list[int]) -> str:
    placeholders = ", ".join(f"${i}" for i in range(len(node_ids)))
    return f"""
MATCH (n)
WHERE id(n) IN [{placeholders}]
RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
       labels(n) AS type, n.name AS name
ORDER BY n.qualified_name
"""


def build_constraint_query(label: str, prop: str) -> str:
    return f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{prop} IS UNIQUE;"


def build_index_query(label: str, prop: str) -> str:
    return f"CREATE INDEX ON :{label}({prop});"


def build_merge_node_query(label: str, id_key: str) -> str:
    return f"MERGE (n:{label} {{{id_key}: row.id}})\nSET n += row.props"


def build_merge_relationship_query(
    from_label: str,
    from_key: str,
    rel_type: str,
    to_label: str,
    to_key: str,
    has_props: bool = False,
) -> str:
    query = (
        f"MATCH (a:{from_label} {{{from_key}: row.from_val}}), "
        f"(b:{to_label} {{{to_key}: row.to_val}})\n"
        f"MERGE (a)-[r:{rel_type}]->(b)\n"
    )
    query += CYPHER_SET_PROPS_RETURN_COUNT if has_props else CYPHER_RETURN_COUNT
    return query


def build_create_node_query(label: str, id_key: str) -> str:
    return f"CREATE (n:{label} {{{id_key}: row.id}})\nSET n += row.props"


def build_create_relationship_query(
    from_label: str,
    from_key: str,
    rel_type: str,
    to_label: str,
    to_key: str,
    has_props: bool = False,
) -> str:
    query = (
        f"MATCH (a:{from_label} {{{from_key}: row.from_val}}), "
        f"(b:{to_label} {{{to_key}: row.to_val}})\n"
        f"CREATE (a)-[r:{rel_type}]->(b)\n"
    )
    query += CYPHER_SET_PROPS_RETURN_COUNT if has_props else CYPHER_RETURN_COUNT
    return query
