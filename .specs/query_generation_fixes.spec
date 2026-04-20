# Query Generation and Tool Execution Fixes

## Overview

This spec addresses two distinct issues identified from query logs:

1. **QG-001**: Invalid Memgraph schema introspection query generation
2. **QG-002**: Missing `python` command in shell allowlist

---

## Issue QG-001: Invalid Schema Introspection Query

### Problem

The LLM generated `CALL schema.info()` when asked to show the graph schema. This procedure does not exist in Memgraph, resulting in query failure:

```
ERROR | codebase_rag.services.llm:generate:281 - [CypherGenerator] Error: LLM did not generate a valid query.
```

Generated query:
```cypher
CALL schema.info()
YIELD node_labels, relationship_types, node_property_types, relationship_property_types
RETURN node_labels, relationship_types, node_property_types, relationship_property_types;
```

### Root Cause Analysis

1. **No schema introspection guidance**: The `GRAPH_SCHEMA_AND_RULES` prompt provides static schema definition but no guidance on how to dynamically query schema
2. **No example queries for schema**: Missing examples for introspection queries in `cypher_queries.py`
3. **LLM hallucination**: LLM assumes `schema.info()` exists (possibly from Neo4j or other graph databases)

### Solution

Add schema introspection query guidance and examples to the prompt system.

#### Implementation Steps

**Step 1**: Add schema introspection example queries to `codebase_rag/cypher_queries.py`:

```python
# Schema introspection queries
CYPHER_SCHEMA_NODE_LABELS = """
MATCH (n)
RETURN DISTINCT labels(n) AS node_labels, count(*) AS count
ORDER BY count DESC
LIMIT 50
"""

CYPHER_SCHEMA_RELATIONSHIP_TYPES = """
MATCH ()-[r]->()
RETURN DISTINCT type(r) AS relationship_type, count(*) AS count
ORDER BY count DESC
LIMIT 50
"""

CYPHER_SCHEMA_NODE_PROPERTIES = """
MATCH (n)
WITH labels(n)[0] AS label, keys(n) AS props
UNWIND props AS prop
RETURN DISTINCT label, prop, count(*) AS occurrences
ORDER BY label, occurrences DESC
LIMIT 100
"""

CYPHER_SCHEMA_COMPLETE = """
// Get all node labels with counts
MATCH (n)
WITH labels(n)[0] AS label, count(*) AS node_count
ORDER BY label
// Get all relationship types with counts
CALL { MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS rel_count }
RETURN label, node_count, collect(DISTINCT rel_type) AS relationship_types
ORDER BY label
"""
```

**Step 2**: Update `codebase_rag/prompts.py` to add schema introspection section:

```python
CYPHER_SCHEMA_INTROSPECTION_RULES = """
**5. Schema Introspection Queries**

When asked about graph schema, node labels, relationship types, or property information:

- **Get all node labels with counts**:
  ```cypher
  MATCH (n)
  RETURN DISTINCT labels(n) AS node_labels, count(*) AS count
  ORDER BY count DESC
  LIMIT 50
  ```

- **Get all relationship types with counts**:
  ```cypher
  MATCH ()-[r]->()
  RETURN DISTINCT type(r) AS relationship_type, count(*) AS count
  ORDER BY count DESC
  LIMIT 50
  ```

- **Get properties for a specific node type**:
  ```cypher
  MATCH (n:Function)
  RETURN DISTINCT keys(n) AS properties
  LIMIT 1
  ```

- **IMPORTANT**: Do NOT use `CALL schema.info()` or `CALL db.schema.info()` - these procedures do not exist in Memgraph.
- Use `CALL db.schema.visualization()` for visual schema if available.
- Use standard MATCH queries for schema introspection.
"""
```

**Step 3**: Integrate the schema introspection rules into `GRAPH_SCHEMA_AND_RULES` in `prompts.py`:

Modify `build_graph_schema_and_rules()` to include the new rules:

```python
def build_graph_schema_and_rules() -> str:
    return f"""You are an expert AI assistant for analyzing codebases using a **hybrid retrieval system**: a **Memgraph knowledge graph** for structural queries and a **semantic code search engine** for intent-based discovery.

**1. Graph Schema Definition**
The database contains information about a codebase, structured with the following nodes and relationships.

{CODE_GRAPH_SCHEMA_DEFINITION}

{CYPHER_QUERY_RULES}

{CYPHER_SCHEMA_INTROSPECTION_RULES}
"""
```

**Step 4**: Update `CYPHER_SAFE_CALL_PROCEDURES` in `constants.py` if needed (already contains `db.schema`).

### Validation

1. Test that "show me all node labels" generates valid Cypher
2. Test that "what relationship types exist" generates valid Cypher
3. Test that "show complete graph schema" generates valid multi-part query or correct introspection query

### Files Modified

- `codebase_rag/cypher_queries.py` - Add schema introspection query constants
- `codebase_rag/prompts.py` - Add schema introspection guidance to prompts

---

## Issue QG-002: Missing Python Command in Shell Allowlist

### Problem

User attempted to run a Python migration script via shell command:
```
python -c "from codebase_rag.migrations.data_model_migrations import run_migrations; print(run_migrations(dry_run=True))"
```

This failed with:
```
Command 'python' is not in the allowlist. Available commands: awk, cat, cp, cut, echo, find, git, head, ls, mkdir, mv, mypy, pre-commit, pwd, pytest, rg, rm, rmdir, ruff, sed, sort, tail, tee, tr, uniq, uv, wc, xargs
```

### Root Cause Analysis

1. **Security consideration**: `python` allows arbitrary code execution, which is risky
2. **Current allowlist**: Focused on file operations, git, and development tools
3. **Use case**: Running Python scripts for migrations, inspections, and utility tasks

### Solution Options

#### Option A: Add `python` to Allowlist (Recommended)

Add `python` and `python3` to the allowlist with appropriate restrictions.

**Pros**:
- Enables Python script execution
- Consistent with `pytest` and `mypy` already being allowed
- Useful for migrations and utility scripts

**Cons**:
- Security risk if used maliciously
- Need to consider `python -c` arbitrary code execution

#### Option B: Create Dedicated Python Execution Tool

Create a new tool specifically for Python script execution with sandboxing.

**Pros**:
- Better security control
- Can restrict to specific modules/functions

**Cons**:
- More implementation effort
- Less flexible

#### Option C: Use `uv` for Python Execution

Since `uv` is already in the allowlist, use `uv run python` for Python execution.

**Pros**:
- No new allowlist additions needed
- `uv` is already trusted

**Cons**:
- Requires `uv` to be installed
- Extra layer of indirection

### Recommended Implementation (Option A + Security Restrictions)

**CRITICAL SECURITY NOTE**: `python`/`python3` execute arbitrary code and can perform destructive operations (file writes, network calls, shell execution). They must **NOT** be added to `SHELL_READ_ONLY_COMMANDS`.

**Step 1**: Add `python` and `python3` to `SHELL_COMMAND_ALLOWLIST` only in `codebase_rag/config.py`:

```python
SHELL_COMMAND_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ls",
        "rg",
        # ... existing commands ...
        "python",
        "python3",
    }
)
```

**Step 2**: Add Python-specific dangerous patterns to `SHELL_DANGEROUS_PATTERNS_SEGMENT` in `codebase_rag/constants.py`:

The existing patterns at lines 1644-1646 already catch basic `os` imports. Add these additional patterns for comprehensive coverage:

```python
# Additional dangerous Python patterns (add to SHELL_DANGEROUS_PATTERNS_SEGMENT)
(
    (r"python.*-c.*subprocess", "python subprocess module execution"),
    (r"python.*-c.*eval\s*\(", "python arbitrary code evaluation via eval()"),
    (r"python.*-c.*exec\s*\(", "python arbitrary code execution via exec()"),
    (r"python.*-c.*compile\s*\(", "python code compilation"),
    (r"python.*-c.*open\s*\(", "python file open operations"),
    (r"python.*-c.*write", "python file write operations"),
    (r"python.*-c.*urllib", "python network access via urllib"),
    (r"python.*-c.*requests", "python network access via requests"),
    (r"python.*-c.*socket", "python network socket operations"),
    (r"python.*-c.*shutil", "python file operations via shutil"),
    (r"python.*-c.*pathlib.*write", "python file writes via pathlib"),
)
```

### Alternative: Use `uv run` (Recommended)

Since `uv` is already in `SHELL_COMMAND_ALLOWLIST` and `python`/`python3` are NOT in `SHELL_READ_ONLY_COMMANDS`, the preferred approach for running Python scripts is:

```
uv run python -c "..."
```

This provides:
- Environment isolation via `uv`
- No additional allowlist changes required
- Better reproducibility

**Recommendation**: Document `uv run python` as the standard approach for Python script execution. Only add raw `python` to the allowlist if `uv` is unavailable.

### Validation

1. Test `python --version` executes successfully
2. Test `python -c "print('hello')"` executes successfully
3. Test dangerous patterns are blocked
4. Verify migration scripts can be run

### Files Modified

- `codebase_rag/config.py` - Add `python` and `python3` to `SHELL_COMMAND_ALLOWLIST` only
- `codebase_rag/constants.py` - Add Python-specific dangerous patterns to `SHELL_DANGEROUS_PATTERNS_SEGMENT`

---

## Implementation Priority

| Issue | Priority | Effort | Impact |
|-------|----------|--------|--------|
| QG-001 | High | Low | Prevents schema queries from working |
| QG-002 | Medium | Low | Limits script execution capabilities |

## Testing Strategy

### QG-001 Tests

```python
# test_schema_introspection.py

def test_node_labels_query_generation():
    """Test that 'show all node labels' generates valid Cypher."""
    result = cypher_generator.generate("show all node labels")
    assert "MATCH (n)" in result
    assert "labels(n)" in result
    assert "schema.info" not in result.lower()

def test_relationship_types_query_generation():
    """Test that 'show relationship types' generates valid Cypher."""
    result = cypher_generator.generate("show all relationship types")
    assert "MATCH ()-[r]->()" in result
    assert "type(r)" in result
    assert "schema.info" not in result.lower()

def test_complete_schema_query():
    """Test complete schema request generates valid queries."""
    result = cypher_generator.generate("show complete graph schema")
    # Should not contain invalid procedures
    assert "schema.info()" not in result.lower()
```

### QG-002 Tests

```python
# test_shell_python.py

def test_python_in_allowlist():
    """Test that python and python3 are in SHELL_COMMAND_ALLOWLIST."""
    assert "python" in settings.SHELL_COMMAND_ALLOWLIST
    assert "python3" in settings.SHELL_COMMAND_ALLOWLIST

def test_python_not_in_read_only():
    """Test that python is NOT in SHELL_READ_ONLY_COMMANDS (security)."""
    assert "python" not in settings.SHELL_READ_ONLY_COMMANDS
    assert "python3" not in settings.SHELL_READ_ONLY_COMMANDS

def test_python_version_allowed():
    """Test that python --version executes successfully with confirmation."""
    result = shell_commander.execute("python --version")
    # Requires confirmation since python is not read-only
    assert result.return_code == 0 or result.requires_confirmation

def test_python_c_blocked_for_dangerous_patterns():
    """Test that dangerous python patterns are blocked."""
    result = shell_commander.execute("python -c \"import os; os.system('rm -rf /')\"")
    assert "blocked" in result.stderr.lower() or "dangerous" in result.stderr.lower()

def test_python_safe_execution():
    """Test safe python -c execution (print statement)."""
    result = shell_commander.execute("python -c \"print('hello')\"")
    # May require confirmation; if allowed, should succeed
    if result.return_code == 0:
        assert "hello" in result.stdout

def test_uv_run_python_works():
    """Test that uv run python is the preferred approach."""
    result = shell_commander.execute("uv run python --version")
    assert result.return_code == 0
```

---

## Rollback Plan

Both changes are low-risk:

1. **QG-001**: Prompt changes can be reverted by removing the new sections
2. **QG-002**: `python` can be removed from allowlist if security concerns arise

---

## References

- `codebase_rag/prompts.py` - Cypher generation prompts
- `codebase_rag/cypher_queries.py` - Example Cypher queries
- `codebase_rag/config.py` - Shell command allowlist configuration
- `codebase_rag/constants.py` - CYPHER_SAFE_CALL_PROCEDURES
- `codebase_rag/tests/test_cypher_validation.py` - Cypher validation tests
