# Tool Expansion Roadmap — Implementation Specification

## Context

The current agentic tool set (`query_graph`, `read_file`, `create_file`, `replace_code`,
`list_directory`, `analyze_document`, `execute_shell`, `semantic_search`,
`get_function_source`, `get_code_snippet`, plus document GraphRAG tools) covers core
code reading and querying. This spec identifies **additional** tools that provide clear
value beyond what existing tools already deliver.

### Existing tool coverage (what NOT to duplicate)

| Capability | Existing tool |
|-----------|---------------|
| Natural-language graph queries | `query_graph` |
| Semantic function search | `semantic_search` |
| Source code by qualified name | `get_code_snippet` |
| Source code by node ID | `get_function_source` |
| File reading | `read_file` |
| File/directory listing | `list_directory` |
| Code editing | `replace_code`, `create_file` |
| Shell commands (allowlisted) | `execute_shell` |
| Document analysis | `analyze_document` |
| Document graph queries | `query_document_graph`, `query_both_graphs` |

## Proposed New Tools

### Phase 1: Python Introspection (HIGH priority)

See [python_introspection_tool.md](./python_introspection_tool.md) for full spec.

- `inspect_python_object` — runtime introspection of modules, classes, functions

**Rationale**: Currently impossible without expanding the shell allowlist to include
`python3` and `pip`, which is a security regression.

### Phase 2: Graph Navigation (MEDIUM priority)

These query the knowledge graph in ways `query_graph` (natural language) and
`semantic_search` (embeddings) do not efficiently cover — structured, deterministic
queries with precise filtering.

#### 2a. `find_references`

Find all references (callers, importers) to a named entity.

```python
async def find_references(qualified_name: str, reference_type: str = "all") -> str
```

- `reference_type`: `"calls"`, `"imports"`, `"all"`
- Queries CALLS/IMPORTS relationships directly (deterministic, not LLM-generated Cypher)
- Returns formatted list of references with file paths and line numbers

**Value over `query_graph`**: Deterministic graph traversal vs. LLM-generated Cypher.
Faster and more reliable for known entity names.

**Value over `semantic_search`**: Exact name matching vs. embedding similarity.

#### 2b. `get_call_hierarchy`

Get callers and/or callees for a function/method.

```python
async def get_call_hierarchy(qualified_name: str, direction: str = "both", depth: int = 2) -> str
```

- `direction`: `"callers"`, `"callees"`, `"both"`
- Traverses CALLS relationships with depth limit
- Returns hierarchical text tree

**Value over `query_graph`**: Structured traversal with depth control vs. free-form
natural language query.

#### 2c. `find_implementations`

Find all classes implementing an interface or inheriting from a base class.

```python
async def find_implementations(interface_name: str) -> str
```

- Queries IMPLEMENTS/INHERITS relationships
- Returns list of implementing classes with file locations

### Phase 3: Project Understanding (MEDIUM priority)

#### 3a. `get_project_structure`

High-level project overview combining directory listing with graph metadata.

```python
async def get_project_structure() -> str
```

- Combines filesystem listing with graph node counts by type
- Returns formatted overview: key directories, language breakdown, entry points
- Cached (result doesn't change between filesystem/graph mutations)

**Value over `list_directory`**: Enriched with graph metadata (function counts,
class counts per package).

#### 3b. `get_import_dependencies`

Analyze import dependency tree for a module.

```python
async def get_import_dependencies(module_path: str, depth: int = 3) -> str
```

- Traverses IMPORTS relationships with depth limit
- Detects circular dependencies
- Returns formatted dependency tree

### Phase 4: Git Integration (LOW priority)

These are low priority because `execute_shell` already supports `git` subcommands
(status, log, diff, show, ls-files, remote, config, branch).

#### 4a. `get_git_status`

Structured git status (not raw shell output).

```python
async def get_git_status() -> str
```

- Wraps `git status --porcelain` with structured parsing
- Returns branch, staged/unstaged/untracked file lists
- **Marginal value** over `execute_shell("git status")` — justified only if
  the structured format significantly helps LLM comprehension

#### 4b. `get_git_history`

Commit history for a path.

```python
async def get_git_history(path: str | None = None, limit: int = 10) -> str
```

- Wraps `git log` with structured output
- **Marginal value** — same caveat as above

## Tools NOT Proposed (and why)

| Dropped tool | Reason |
|-------------|--------|
| `execute_safe_python` | `exec()`-based sandboxing is fundamentally insecure in Python. The restricted execution model has well-known bypass paths. The `inspect_python_object` tool achieves the same goals safely. |
| `find_similar_code` | Requires a code similarity engine (tree edit distance, code2vec, etc.) — no existing infrastructure. Major new subsystem, not a tool addition. |
| `check_dependency_conflicts` | Requires external vulnerability database APIs (OSV, PyPI JSON API). Network dependency, rate limits, and API stability concerns. Not a tool — an integration. |
| `generate_docstring` / `generate_test_cases` | Require LLM-in-the-loop for generation. These are orchestrator features, not standalone tools. They need access to the app's LLM provider and token budget management. |
| `regex_search` | `execute_shell` already allows `rg` (ripgrep), which is faster and more capable than any Python reimplementation. |
| `batch_file_operations` | Composition of existing tools (`read_file`, `copy`, `move`). No new capability. |
| `execute_cypher_query` | Direct Cypher execution is dangerous (write queries can corrupt data). The existing `query_graph` tool uses LLM-generated Cypher with safety constraints. |

## Implementation Pattern

All new tools follow the established factory pattern:

```python
# tools/new_tool.py
from __future__ import annotations

from loguru import logger
from pydantic_ai import Tool

from .. import tool_errors as te
from . import tool_descriptions as td


class NewToolClass:
    __slots__ = ("project_root",)

    def __init__(self, project_root: str = "."):
        from pathlib import Path
        self.project_root = Path(project_root).resolve()
        if self.project_root.is_file():
            self.project_root = self.project_root.parent


def create_new_tool(instance: NewToolClass) -> Tool:
    async def new_tool_function(param: str) -> str:
        result = await instance.do_something(param)
        if result.error_message:
            return te.ERROR_WRAPPER.format(message=result.error_message)
        return format_result(result)

    return Tool(
        function=new_tool_function,
        name=td.AgenticToolName.NEW_TOOL,
        description=td.NEW_TOOL_DESC,
    )
```

### Registration checklist for each new tool

1. **Schema** → `schemas.py` — add result Pydantic model
2. **Module** → `tools/<tool_name>.py` — implement class + factory function
3. **Enum** → `tools/tool_descriptions.py` — add `AgenticToolName` entry
4. **Description** → `tools/tool_descriptions.py` — add description constant + `AGENTIC_TOOLS` entry
5. **MCP** → `constants.py` — add `MCPToolName` entry (if MCP-compatible)
6. **MCP desc** → `tools/tool_descriptions.py` — add MCP description + `MCP_TOOLS` entry
7. **Logs** → `logs.py` — add log format constants
8. **Errors** → `tool_errors.py` — add error format constants
9. **Wire up** → `main.py` — import, instantiate, append to `tools` list
10. **Tests** → `tests/test_<tool_name>.py` — unit + integration tests

## Configuration

New config fields (if needed) go into `AppConfig` in `config.py`:

```python
# Example: timeout for introspection operations
PYTHON_INSPECT_TIMEOUT: int = Field(default=10, gt=0)
```

Follow the existing pattern: `Field(default=..., validators)`, env var binding via
`pydantic_settings`, and property access through `settings`.

## Priority Order

1. **Phase 1**: `inspect_python_object` — unblocks current shell-allowlist limitation
2. **Phase 2**: `find_references`, `get_call_hierarchy`, `find_implementations` —
   deterministic graph traversal complements LLM-based `query_graph`
3. **Phase 3**: `get_project_structure`, `get_import_dependencies` — project overview
4. **Phase 4**: Git tools — low priority since `execute_shell` already supports git
