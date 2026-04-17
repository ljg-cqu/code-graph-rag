---
description: "Complete CLI reference for Code-Graph-RAG commands and Makefile targets."
---

# CLI Reference

The `cgr` command is the main entry point for Code-Graph-RAG.

## Global Options

These options apply to all commands:

| Option | Description |
|--------|-------------|
| `-v/--version` | Show version number and exit |
| `-q/--quiet` | Suppress non-essential output (progress messages, banners, informational logs) |

## Core Commands

### `cgr start`

Parse a repository and/or start the interactive query CLI.

```bash
cgr start --repo-path /path/to/repo [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--repo-path` | Path to repository (defaults to current directory) |
| `--index-code` | Index code assets from repository into the knowledge graph |
| `--clean` | Clear existing data before ingesting |
| `--batch-size` | Override Memgraph flush batch size |
| `--orchestrator` | Specify provider:model for main operations (e.g., `google:gemini-2.5-pro`, `ollama:llama3.2`) |
| `--cypher` | Specify provider:model for graph queries (e.g., `google:gemini-2.5-flash`, `ollama:codellama`) |
| `-o/--output` | Export graph to JSON file during update |
| `--no-confirm` | Suppress confirmation prompts |
| `-y/--yolo` | Alias for --no-confirm, skip all confirmation prompts |
| `--project-name` | Override auto-detected project name |
| `--exclude` | Exclude paths matching given pattern from indexing (can be used multiple times) |
| `--interactive-setup` | Interactive setup to select which directories to include/exclude |
| `-a/--ask-agent` | Run a single query directly without entering interactive mode |
| `--with-docs` | Connect to document graph for dual-graph querying |
| `--index-docs` | Index documents before starting chat |
| `--index-all` | Index both code and documents before starting chat |
| `--doc-workspace` | Document graph workspace identifier (default: `default`) |
| `--check-freshness` | Check if graphs are up-to-date before starting (default: enabled, use `--no-check-freshness` to disable) |
| `--mode` | Query routing mode: `code_only`, `document_only`, `both_merged`, `code_vs_doc`, `doc_vs_code` |
| `--index-timeout` | Maximum seconds for indexing operations (default: 300s) |

#### Unified Dual-Graph Usage

Query both code and documents from a single session:

```bash
# Index and query both code and documents
cgr start --repo-path /path/to/repo --index-all --with-docs --mode both_merged

# Start with document graph for spec validation
cgr start --repo-path /path/to/repo --with-docs --mode code_vs_doc

# Index documents only, then query
cgr start --repo-path /path/to/repo --index-docs --with-docs --mode document_only
```

#### In-Chat Commands

| Command | Description |
|---------|-------------|
| `/mode <mode>` | Switch query mode during chat (e.g., `/mode both_merged`) |
| `/model <provider:model>` | Switch LLM model during chat |
| `/models` | Show all available models (static catalog + .env configured) with configuration status |
| `/help` | Show available commands |
| `/exit` | Exit the session |

#### Query Modes

| Mode | Description |
|------|-------------|
| `code_only` | Query code graph only (default) |
| `document_only` | Query document graph only |
| `both_merged` | Query both graphs, merge results with attribution |
| `code_vs_doc` | Validate code against documentation (docs = truth) |
| `doc_vs_code` | Validate documentation against code (code = truth) |

### `cgr export`

Export the knowledge graph to JSON.

```bash
cgr export -o my_graph.json [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-o/--output` | Path to output JSON file (required) |
| `--batch-size` | Override Memgraph flush batch size |
| `--json/--no-json` | Output format (only JSON is currently supported) |

### `cgr ingest-json`

Ingest custom entities, relationships, and domain knowledge from JSON files directly into the knowledge graph and vector database.

```bash
cgr ingest-json --input-path /path/to/input [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--input-path` | Path to single JSON file or directory containing multiple JSON files (required) |
| `--dataset-id` | Optional custom identifier for the dataset (defaults to value defined in JSON metadata) |
| `--skip-existing` | Skip entities/relationships that already exist in the graph |
| `--incremental` | Run incremental update, only process changed entities/relationships (uses last modified timestamps) |
| `--dry-run` | Validate input and calculate changes without writing to databases |
| `--conflict-resolution` | Conflict resolution strategy for overlapping updates: `last-write-wins` (default), `highest-confidence-wins`, `manual-review` |
| `--batch-size` | Override default batch size for database operations |

### `cgr optimize`

AI-powered codebase optimization.

```bash
cgr optimize <language> --repo-path /path/to/repo [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--repo-path` | Path to repository |
| `--orchestrator` | Specify provider:model for main operations |
| `--cypher` | Specify provider:model for graph queries |
| `--batch-size` | Override Memgraph flush batch size |
| `--reference-document` | Path to reference documentation for guided optimization |
| `--no-confirm` | Suppress confirmation prompts |
| `-y/--yolo` | Alias for --no-confirm, skip all confirmation prompts |

Supported languages: `python`, `javascript`, `typescript`, `rust`, `go`, `java`, `scala`, `cpp`

### `cgr mcp-server`

Start the MCP server for Claude Code integration.

```bash
cgr mcp-server [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--transport` | Transport mode: `stdio` (default) or `http` |
| `--host` | HTTP server host (default: 127.0.0.1) |
| `--port` | HTTP server port (default: 8000) |

### `cgr index`

Index a repository to protobuf for offline use.

```bash
cgr index -o ./index-output --repo-path ./my-project [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-o/--output-proto-dir` | Path to output directory for protobuf files (required) |
| `--repo-path` | Path to repository (defaults to current directory) |
| `--split-index` | Split index into multiple files for large repositories |
| `--exclude` | Exclude paths matching given pattern from indexing (can be used multiple times) |
| `--interactive-setup` | Interactive setup to select which directories to include/exclude |

### `cgr doctor`

Check that all required dependencies and services are available.

```bash
cgr doctor
```

### `cgr stats`

Show statistics about the code graph (node counts, relationship counts, etc.).

```bash
cgr stats
```

### `cgr graph-loader`

Load an exported graph file and display summary information.

```bash
cgr graph-loader <graph-file-path>
```

### `cgr language`

Manage language support.

```bash
cgr language add-grammar <language-name>
cgr language add-grammar --grammar-url <url>
cgr language list-languages
cgr language remove-language <language-name>
```

## Document GraphRAG Commands

### `cgr index-docs`

Index documents (Markdown, PDF, DOCX) into the document graph.

```bash
cgr index-docs --repo-path /path/to/repo [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--repo-path` | Path to repository (defaults to current directory) |
| `--clean` | Clear document database before indexing |
| `--force` | Force re-indexing (ignore version cache) |

### `cgr query-docs`

Query the document graph using natural language.

```bash
cgr query-docs "How do I use the API?" [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--top-k` | Number of results to return (default: 5) |

### `cgr query-all`

Query both code and document graphs with merged results.

```bash
cgr query-all "Tell me about authentication" [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--repo-path` | Path to repository (defaults to current directory) |
| `--top-k` | Number of results per graph (default: 5) |

### `cgr validate-spec`

Validate code against a specification document.

```bash
cgr validate-spec <spec-path> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--repo-path` | Path to repository (defaults to current directory) |
| `--scope` | Validation scope: `all`, `sections`, or `claims` (default: `all`) |
| `--max-cost` | Maximum cost budget in USD (default: 0.50) |
| `--dry-run` | Estimate cost without running validation |

### `cgr validate-doc`

Validate documentation against actual code.

```bash
cgr validate-doc <doc-path> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--repo-path` | Path to repository (defaults to current directory) |
| `--scope` | Validation scope: `all`, `sections`, or `claims` (default: `all`) |
| `--max-cost` | Maximum cost budget in USD (default: 0.50) |
| `--dry-run` | Estimate cost without running validation |

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show help message |
| `make all` | Install everything for full development environment |
| `make install` | Install project dependencies with full language support |
| `make python` | Install project dependencies for Python only |
| `make dev` | Setup development environment (install deps + pre-commit hooks) |
| `make test` | Run unit tests only (fast, no Docker) |
| `make test-parallel` | Run unit tests in parallel (fast, no Docker) |
| `make test-integration` | Run integration tests (requires Docker) |
| `make test-all` | Run all tests including integration and e2e (requires Docker) |
| `make test-parallel-all` | Run all tests in parallel (requires Docker) |
| `make clean` | Clean up build artifacts and cache |
| `make build-grammars` | Build grammar submodules |
| `make watch` | Watch repository for changes and update graph in real-time |
| `make readme` | Regenerate README.md from codebase |
| `make lint` | Run ruff check |
| `make format` | Run ruff format |
| `make typecheck` | Run type checking with ty |
| `make check` | Run all checks: lint, typecheck, test |
| `make pre-commit` | Run all pre-commit checks locally |
