<div align="center">
  <picture>
    <source srcset="assets/logo-dark-any.png" media="(prefers-color-scheme: dark)">
    <source srcset="assets/logo-light-any.png" media="(prefers-color-scheme: light)">
    <img src="assets/logo-dark-any.png" alt="Code-Graph-RAG Logo" width="480">
  </picture>

  <p>
  <a href="https://github.com/vitali87/code-graph-rag/stargazers">
    <img src="https://img.shields.io/github/stars/vitali87/code-graph-rag?style=social" alt="GitHub stars" />
  </a>
  <a href="https://github.com/vitali87/code-graph-rag/network/members">
    <img src="https://img.shields.io/github/forks/vitali87/code-graph-rag?style=social" alt="GitHub forks" />
  </a>
  <a href="https://codecov.io/gh/vitali87/code-graph-rag">
    <img src="https://codecov.io/gh/vitali87/code-graph-rag/graph/badge.svg" alt="Codecov" />
  </a>
  <a href="https://sonarcloud.io/summary/overall?id=vitali87_code-graph-rag">
    <img src="https://sonarcloud.io/api/project_badges/measure?project=vitali87_code-graph-rag&metric=alert_status" alt="Quality Gate Status" />
  </a>
  <a href="https://mseep.ai/app/vitali87-code-graph-rag">
    <img src="https://mseep.net/pr/vitali87-code-graph-rag-badge.png" alt="MseeP.ai Security Assessment" height="20" />
  </a>
  <a href="https://code-graph-rag.com">
    <img src="https://img.shields.io/badge/Enterprise-Support%20%26%20Services-6366f1" alt="Enterprise Support" />
  </a>
  <a href="https://pepy.tech/projects/code-graph-rag">
    <img src="https://static.pepy.tech/personalized-badge/code-graph-rag?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" />
  </a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/vitali87/code-graph-rag">
    <img src="https://api.scorecard.dev/projects/github.com/vitali87/code-graph-rag/badge" alt="OpenSSF Scorecard" />
  </a>
  <a href="https://gitcgr.com/vitali87/code-graph-rag">
    <img src="https://gitcgr.com/badge/vitali87/code-graph-rag.svg" alt="gitcgr" />
  </a>
</p>
</div>

# Code-Graph-RAG: A Graph-Based RAG System for Any Codebases

An accurate Retrieval-Augmented Generation (RAG) system that analyzes multi-language codebases using Tree-sitter, builds comprehensive knowledge graphs, and enables natural language querying of codebase structure and relationships as well as editing capabilities.

**🔥 New: Document GraphRAG Support** — Index and query documentation (Markdown, PDF, DOCX) alongside your code. Validate code against specifications, detect outdated documentation, and get comprehensive answers spanning both code and docs.


<p align="center">
  <img src="./assets/demo.gif" alt="demo">
</p>

## Latest News 🔥

- **📄 Automatic JSON Ingestion on Start**: Automatically ingest valid JSON data when running `cgr start --index-docs` or `--index-all`, with parallel processing (up to 32 workers), automatic schema validation against [ingestion_schema.json](./ingestion_schema.json), and workspace isolation.
- **🌐 Global Filesystem Access**: Full support for reading, writing, and editing files anywhere on the host filesystem, with configurable security controls and approval workflows.
- **📚 Document GraphRAG Support**: Full document indexing and querying now available! Index Markdown, PDF, DOCX files and query them alongside your code. Features include bidirectional validation (code vs docs), merged queries across both graphs, and specification compliance checking.
- **💎 Solidity Support**: Full Solidity smart contract support added — contracts, interfaces, libraries, events, modifiers, state variables, fallback/receive functions, and call graph analysis for blockchain development.
- **🤖 AutoHotkey Support**: AutoHotkey (AHK) scripting language support added — functions, labels, hotkeys, hotstrings, and include directive parsing. Perfect for Windows automation script analysis.
- **PHP Language Support**: Full PHP language support added — classes, interfaces, traits, enums, namespaces, PHP 8 attributes, and call graph analysis. Contributed by [@rs-ipps](https://github.com/rs-ipps).
- **C Language Support**: Full C language support added — functions, structs, unions, enums, preprocessor includes, and call graph analysis. Contributed by [@dj0nes](https://github.com/dj0nes).
- **Visualise any GitHub repo instantly!** Just change `github.com` to `gitcgr.com` in any repo URL — that's it, only 3 letters! Get an interactive graph of the entire codebase structure. Try it now: [gitcgr.com](https://gitcgr.com)

## 🚀 Features

- **⚡ Memgraph Native Vector Storage**: No external vector database required. All embeddings are stored directly on graph nodes for atomic hybrid vector+graph queries, eliminating cross-service latency and data duplication.
- **🧠 Graph Algorithm Integration**: Automatic post-ingestion algorithm runs improve retrieval relevance:
  - PageRank calculation identifies important code entities (core classes, frequently called functions) for better ranking
  - Leiden/Louvain community detection groups related code entities for global architecture analysis
  - BFS context expansion automatically retrieves related code context during search
- **⚡ Automatic Parallel Execution**: No explicit user request needed for safe read-only work. The system can preview path-scoped subtasks, block write-like requests from parallel execution, run real read-only sub-agents in parallel, and fall back to sequential execution when the task is unsafe or underspecified.
- **⚡ Blazing Fast Parallel Indexing**: Up to 20x faster codebase ingestion with perfect round-robin load distribution across parallel workers, auto-optimized at runtime to match your CPU core count and workload size (never uses more workers than needed, no wasted overhead). Fully backward compatible with sequential mode, no configuration required out of the box.
- **🔄 Integrated Realtime Updates**: Keep your knowledge graph synchronized automatically with the `--realtime-updater` flag. Watches for file changes in the background while you chat, updating the graph instantly for code, documents, and JSON files. No separate terminal needed—shares database connections with the chat session for efficiency.
- **🧠 Intelligent Context Window Compression**: Automatically prevents LLM context window overflow with zero semantic loss for critical content, no manual intervention required. Uses 10 parallel round-robin workers (supports up to 20 for high throughput workloads) to evaluate 5 compression strategies and select the optimal one per scenario using weighted scoring (60% semantic retention, 30% token reduction, 10% execution speed). Features include: automatic 85% usage trigger with 5% hysteresis buffer, manual `/compress` CLI command with aggressive mode and custom preserve patterns, 24h context archive for restore capability, automatic rollback if retention falls below 70% threshold, and guaranteed preservation of latest 2 user turns, all system prompts, and tool call history. Delivers average 40% token reduction with >88% semantic retention in <200ms per compression run.
- **🧠 Context Window Management System**: Flexible, multi-level context window configuration with automatic model detection:
  - Default context window increased to 256k tokens to align with modern LLM capabilities
  - Support for role-specific overrides (orchestrator, cypher) with highest precedence
  - Support for provider/model-specific overrides via environment variables
  - Automatic context window detection for all common LLM models across supported providers
  - Graceful fallback to global default value when detection fails
  - Fully backwards compatible with existing configurations
- **🤖 Dynamic Model Catalog**: Automatically discovers models configured via `.env` variables (ORCHESTRATOR_*, CYPHER_*, CGR_WORKER_LLMS) and displays them in the `/models` command with configuration status indicators. Supports external catalog files via `CGR_MODEL_CATALOG_PATH` and dynamic discovery toggle via `CGR_DISABLE_MODEL_DISCOVERY`.
- **🔍 Hybrid Retrieval Pipeline**: Combines semantic vector search, graph traversal, and PageRank ranking in a single atomic Memgraph query for more relevant results and lower latency. No separate vector search and graph query steps needed.
- **🌐 Global Filesystem Access**: Access files anywhere on your host system (enabled by default), with optional write approval requirements and built-in protection against path traversal attacks.
- **📚 Document GraphRAG**: Index and query documentation (Markdown, PDF, DOCX) alongside code. Supports bidirectional validation between code and specs, merged queries across both graphs, and automated documentation audits.
- **Multi-Language Support**:

<!-- SECTION:supported_languages -->
| Language | Status | Extensions | Functions | Classes/Structs | Modules | Package Detection | Additional Features |
|--------|------|----------|---------|---------------|-------|-----------------|-------------------|
| AutoHotkey | Fully Supported | .ahk | ✓ | ✓ | ✓ | - | Hotkeys, hotstrings, labels, GUI controls, commands, v1/v2 detection |
| C | Fully Supported | .c | ✓ | ✓ | ✓ | ✓ | Functions, structs, unions, enums, preprocessor includes |
| C++ | Fully Supported | .cpp, .h, .hpp, .cc, .cxx, .hxx, .hh, .ixx, .cppm, .ccm | ✓ | ✓ | ✓ | ✓ | Constructors, destructors, operator overloading, templates, lambdas, C++20 modules, namespaces |
| Java | Fully Supported | .java | ✓ | ✓ | ✓ | - | Generics, annotations, modern features (records/sealed classes), concurrency, reflection |
| JavaScript | Fully Supported | .js, .jsx | ✓ | ✓ | ✓ | - | ES6 modules, CommonJS, prototype methods, object methods, arrow functions |
| Lua | Fully Supported | .lua | ✓ | - | ✓ | - | Local/global functions, metatables, closures, coroutines |
| PHP | Fully Supported | .php | ✓ | ✓ | ✓ | - | Classes, interfaces, traits, enums, namespaces, PHP 8 attributes |
| Python | Fully Supported | .py | ✓ | ✓ | ✓ | ✓ | Type inference, decorators, nested functions |
| Rust | Fully Supported | .rs | ✓ | ✓ | ✓ | ✓ | impl blocks, associated functions |
| Solidity | Fully Supported | .sol | ✓ | ✓ | ✓ | ✓ | Contracts, interfaces, libraries, events, modifiers, state variables, fallback/receive functions |
| TypeScript | Fully Supported | .ts, .tsx | ✓ | ✓ | ✓ | - | Interfaces, type aliases, enums, namespaces, ES6/CommonJS modules |
| C# | In Development | .cs | ✓ | ✓ | ✓ | - | Classes, interfaces, generics (planned) |
| Go | In Development | .go | ✓ | ✓ | ✓ | - | Methods, type declarations |
| Scala | In Development | .scala, .sc | ✓ | ✓ | ✓ | - | Case classes, objects |
<!-- /SECTION:supported_languages -->
- **🌳 Tree-sitter Parsing**: Uses Tree-sitter for robust, language-agnostic AST parsing
- **📊 Knowledge Graph Storage**: Uses Memgraph to store codebase structure as an interconnected graph
- **🗣️ Natural Language Querying**: Ask questions about your codebase in plain English
- **🤖 AI-Powered Cypher Generation**: Supports cloud models (Google Gemini, Anthropic Claude), local models (Ollama), and OpenAI models for natural language to Cypher translation
- **🤖 OpenAI Integration**: Leverage OpenAI models to enhance AI functionalities.
- **📝 Code Snippet Retrieval**: Retrieves actual source code snippets for found functions/methods
- **✍️ Advanced File Editing**: Surgical code replacement with AST-based function targeting, visual diff previews, and exact code block modifications
- **⚡️ Shell Command Execution**: Can execute terminal commands for tasks like running tests or using CLI tools.
- **🔍 Python Introspection**: Inspect Python modules, classes, and functions at runtime without shell access.
- **🗺️ Graph Navigation**: Deterministic graph queries for references, call hierarchies, implementations, and import dependencies.
- **🚀 Interactive Code Optimization**: AI-powered codebase optimization with language-specific best practices and interactive approval workflow
- **📚 Reference-Guided Optimization**: Use your own coding standards and architectural documents to guide optimization suggestions
- **🔗 Dependency Analysis**: Parses `pyproject.toml` to understand external dependencies
- **🎯 Nested Function Support**: Handles complex nested functions and class hierarchies
- **🔄 Language-Agnostic Design**: Unified graph schema across all supported languages
- **📱 Smart Contract Analysis**: Full Solidity support for blockchain development — contracts, events, modifiers, state variables
- **🤖 Automation Script Analysis**: AutoHotkey support for Windows automation — hotkeys, hotstrings, labels, GUI controls

## 🏗️ Architecture

The system consists of two main components:

1. **Multi-language Parser**: Tree-sitter based parsing system that analyzes codebases and ingests data into Memgraph
2. **RAG System** (`codebase_rag/`): Interactive CLI for querying the stored knowledge graph


## 📋 Prerequisites

- Python 3.12+
- Docker & Docker Compose (for Memgraph)
- **cmake** (required for building pymgclient dependency)
- **ripgrep** (`rg`) (required for shell command text searching)
- **For cloud models**: Google Gemini API key
- **For local models**: Ollama installed and running
- `uv` package manager

### Installing cmake and ripgrep

On macOS:
```bash
brew install cmake ripgrep
```

On Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install cmake ripgrep
```

On Linux (CentOS/RHEL):
```bash
sudo yum install cmake
sudo dnf install ripgrep
# Note: ripgrep may need to be installed from EPEL or via cargo
```

## 🛠️ Installation

```bash
git clone https://github.com/vitali87/code-graph-rag.git

cd code-graph-rag
```

2. **Install dependencies**:

For basic Python support:
```bash
uv sync
```

For full multi-language support:
```bash
uv sync --extra treesitter-full
```

For development (including tests and pre-commit hooks):
```bash
make dev
```

This installs all dependencies and sets up pre-commit hooks automatically.

This installs Tree-sitter grammars for all supported languages (see Multi-Language Support section).

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration (see options below)
# ⚠️ Security Note: Always set database credentials (MEMGRAPH_*)
# for production deployments. Never commit your .env file with secrets to version control.
```

### Configuration Options

The new provider-explicit configuration supports mixing different providers for orchestrator and cypher models.

#### Option 1: All Ollama (Local Models)

```bash
# .env file
ORCHESTRATOR_PROVIDER=ollama
ORCHESTRATOR_MODEL=llama3.2
ORCHESTRATOR_ENDPOINT=http://localhost:11434/v1

CYPHER_PROVIDER=ollama
CYPHER_MODEL=codellama
CYPHER_ENDPOINT=http://localhost:11434/v1
```

#### Option 2: All OpenAI Models
```bash
# .env file
ORCHESTRATOR_PROVIDER=openai
ORCHESTRATOR_MODEL=gpt-4o
ORCHESTRATOR_API_KEY=sk-your-openai-key

CYPHER_PROVIDER=openai
CYPHER_MODEL=gpt-4o-mini
CYPHER_API_KEY=sk-your-openai-key
```

#### Option 3: All Google Models
```bash
# .env file
ORCHESTRATOR_PROVIDER=google
ORCHESTRATOR_MODEL=gemini-2.5-pro
ORCHESTRATOR_API_KEY=your-google-api-key

CYPHER_PROVIDER=google
CYPHER_MODEL=gemini-2.5-flash
CYPHER_API_KEY=your-google-api-key
```

#### Option 4: All Anthropic Models
```bash
# .env file
ORCHESTRATOR_PROVIDER=anthropic
ORCHESTRATOR_MODEL=claude-3-5-sonnet-latest
ORCHESTRATOR_API_KEY=sk-ant-api03-your-anthropic-key

CYPHER_PROVIDER=anthropic
CYPHER_MODEL=claude-3-haiku-latest
CYPHER_API_KEY=sk-ant-api03-your-anthropic-key
```

#### Option 5: Anthropic-Compatible Endpoints
```bash
# .env file - Custom Anthropic-compatible endpoint (e.g., Baidu Qianfan)
ORCHESTRATOR_PROVIDER=anthropic
ORCHESTRATOR_MODEL=GLM-5
ORCHESTRATOR_API_KEY=your-custom-api-key
ORCHESTRATOR_ENDPOINT=https://qianfan.baidubce.com/anthropic/coding
```

#### Option 6: Mixed Providers
```bash
# .env file - Google orchestrator + Ollama cypher
ORCHESTRATOR_PROVIDER=google
ORCHESTRATOR_MODEL=gemini-2.5-pro
ORCHESTRATOR_API_KEY=your-google-api-key

CYPHER_PROVIDER=ollama
CYPHER_MODEL=codellama
CYPHER_ENDPOINT=http://localhost:11434/v1
```

Get your Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

#### Context Window Configuration Options
The system provides flexible context window management with multiple configuration levels (highest to lowest precedence):
1. **Role-specific overrides**: Set custom context window sizes for orchestrator and cypher models:
   ```bash
   ORCHESTRATOR_CONTEXT_WINDOW=256000
   CYPHER_CONTEXT_WINDOW=128000
   ```
2. **Provider/model-specific overrides**: Set context window for specific models. Model ID is normalized to uppercase with spaces/hyphens/periods replaced with underscores:
   ```bash
   # Examples:
   OPENAI_GPT_4O_CONTEXT_WINDOW=128000
   ANTHROPIC_CLAUDE_3_5_SONNET_CONTEXT_WINDOW=200000
   GOOGLE_GEMINI_2_5_PRO_CONTEXT_WINDOW=1048576
   OLLAMA_LLAMA3_1_CONTEXT_WINDOW=128000
   ```
3. **Automatic detection**: The system automatically detects context window sizes for all common LLM models across supported providers
4. **Global default**: Fallback value when no other configuration is found (default: 256000 tokens):
   ```bash
   DEFAULT_CONTEXT_WINDOW=256000
   ```

All new configuration options are fully backwards compatible. To keep the old 128k default, set `DEFAULT_CONTEXT_WINDOW=128000` in your `.env` file.

**Install and run Ollama**:
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2
# Or try other models like:
# ollama pull llama3
# ollama pull mistral
# ollama pull codellama

# Ollama will automatically start serving on localhost:11434
```

> **Note**: Local models provide privacy and no API costs, but may have lower accuracy compared to cloud models like Gemini.

4. **Start Memgraph database**:
```bash
docker compose up -d
```

5. **Verify installation**:
```bash
# If installed from PyPI:
cgr --help

# If running from source:
uv run cgr --help
```

> **Note**: When running from source (cloned repo), prefix all `cgr` commands below with `uv run`, e.g., `uv run cgr start ...`

## 🛠️ Makefile Commands

Use the Makefile for common development tasks:

<!-- SECTION:makefile_commands -->
| Command | Description |
|-------|-----------|
| `make help` | Show this help message |
| `make all` | Install everything for full development environment (deps, grammars, hooks, tests) |
| `make install` | Install project dependencies with full language support |
| `make python` | Install project dependencies for Python only |
| `make dev` | Setup development environment (install deps + pre-commit hooks) |
| `make test` | Run unit tests only (fast, no Docker) |
| `make test-parallel` | Run unit tests in parallel (fast, no Docker) |
| `make test-integration` | Run integration tests (requires Docker) |
| `make test-all` | Run all tests including integration and e2e (requires Docker) |
| `make test-parallel-all` | Run all tests in parallel including integration and e2e (requires Docker) |
| `make clean` | Clean up build artifacts and cache |
| `make build-grammars` | Build grammar submodules |
| `make watch` | Watch repository for changes and update graph in real-time |
| `make readme` | Regenerate README.md from codebase |
| `make lint` | Run ruff check |
| `make format` | Run ruff format |
| `make typecheck` | Run type checking with ty |
| `make check` | Run all checks: lint, typecheck, test |
| `make pre-commit` | Run all pre-commit checks locally (comprehensive test before commit) |
<!-- /SECTION:makefile_commands -->

## 🎯 Usage

The Code-Graph-RAG system offers seven main modes of operation:
1. **Parse & Ingest**: Build knowledge graph from your codebase
2. **Interactive Query**: Ask questions about your code in natural language
3. **Export & Analyze**: Export graph data for programmatic analysis
4. **Editing**: Perform surgical code replacements and modifications with precise targeting.
5. **📚 Document GraphRAG**: Index and query documentation alongside code, with validation capabilities.
6. **📄 JSON Data Ingestion**: Ingest custom entities, relationships, and domain knowledge directly from JSON files
7. **AI Optimization**: Get AI-powered optimization suggestions for your code.

### Step 1: Parse a Repository

Parse and ingest a multi-language repository into the knowledge graph:

**For the first repository (clean start):**
```bash
cgr start --repo-path /path/to/repo1 --index-code --clean
```

**For additional repositories (preserve existing data):**
```bash
cgr start --repo-path /path/to/repo2 --index-code
cgr start --repo-path /path/to/repo3 --index-code
```

**Control Memgraph batch flushing:**
```bash
# Flush every 5,000 records instead of the default from settings
cgr start --repo-path /path/to/repo --index-code \
  --batch-size 5000
```

**Control parallel indexing worker count:**
```bash
# Override default 20 parallel workers with 15
cgr start --repo-path /path/to/repo --index-code \
  --parallel-workers 15

# Force sequential indexing (disable parallelism entirely)
cgr start --repo-path /path/to/repo --index-code \
  --parallel-workers 1
```

> **💡 Auto-Optimization Note**: The system automatically adjusts worker count at runtime:
> - Never uses more workers than available CPU cores (prevents thrashing)
> - Never uses more workers than number of changed files (avoids wasted process startup overhead)
> - Uses perfect round-robin distribution of files across workers for balanced load

The system automatically detects and processes files for all supported languages (see Multi-Language Support section).

### Step 2: Query the Codebase

Start the interactive RAG CLI:

```bash
cgr start --repo-path /path/to/your/repo
```

### Step 2.5: Real-Time Graph Updates (Optional)

For active development, you can keep your knowledge graph automatically synchronized with code changes using the realtime updater. This is particularly useful when you're actively modifying code and want the AI assistant to always work with the latest codebase structure.

**What it does:**
- Watches your repository for file changes (create, modify, delete)
- Automatically updates the knowledge graph in real-time
- Maintains consistency by recalculating all function call relationships
- Filters out irrelevant files (`.git`, `node_modules`, etc.)
- Supports code files, documents, and JSON files simultaneously

**Two Ways to Use:**

#### Option 1: Integrated Realtime Updater (Recommended)

The easiest way - just add `--realtime-updater` to your `cgr start` command:

```bash
# Enable realtime updates for code files only (default)
cgr start --repo-path /path/to/your/repo --realtime-updater

# Enable realtime updates for code and documents
cgr start --repo-path /path/to/your/repo --realtime-updater --realtime-docs --with-docs

# Enable realtime updates for all file types (code, docs, JSON)
cgr start --repo-path /path/to/your/repo --realtime-updater --realtime-docs --realtime-json --with-docs

# Customize debounce timing (wait 2s after last change, max 10s wait)
cgr start --repo-path /path/to/your/repo --realtime-updater --realtime-debounce 2 --realtime-max-wait 10
```

**Benefits of integrated mode:**
- Single command to start both chat and file watcher
- Shares database connections (more efficient)
- Same configuration for chat and updater
- Graceful shutdown when you exit the chat

#### Option 2: Standalone Realtime Updater (Legacy)

Run the realtime updater in a separate terminal for more control:

```bash
# Terminal 1: Start the realtime updater
python realtime_updater.py /path/to/your/repo

# Terminal 2: Run the AI assistant
cgr start --repo-path /path/to/your/repo
```

**With custom Memgraph settings:**
```bash
# Python
python realtime_updater.py /path/to/your/repo --host localhost --port 7687 --batch-size 1000

# Makefile
make watch REPO_PATH=/path/to/your/repo HOST=localhost PORT=7687 BATCH_SIZE=1000
```

**Performance note:** The updater currently recalculates all CALLS relationships on every file change to ensure consistency. This prevents "island" problems where changes in one file aren't reflected in relationships from other files, but may impact performance on very large codebases with frequent changes. **Note:** Optimization of this behavior is a work in progress.

**Standalone CLI Arguments:**
- `repo_path` (required): Path to repository to watch
- `--host`: Memgraph host (default: `localhost`)
- `--port`: Memgraph port (default: `7687`)
- `--batch-size`: Number of buffered nodes/relationships before flushing to Memgraph
- `--debounce`: Debounce delay in seconds (default: `5.0`)
- `--max-wait`: Maximum wait time before processing (default: `30.0`)

**Specify Custom Models:**
```bash
# Use specific local models
cgr start --repo-path /path/to/your/repo \
  --orchestrator ollama:llama3.2 \
  --cypher ollama:codellama

# Use specific Gemini models
cgr start --repo-path /path/to/your/repo \
  --orchestrator google:gemini-2.0-flash-thinking-exp-01-21 \
  --cypher google:gemini-2.5-flash-lite-preview-06-17

# Use mixed providers
cgr start --repo-path /path/to/your/repo \
  --orchestrator google:gemini-2.0-flash-thinking-exp-01-21 \
  --cypher ollama:codellama
```

Example queries (works across all supported languages):
- "Show me all classes that contain 'user' in their name"
- "Find functions related to database operations"
- "What methods does the User class have?"
- "Show me functions that handle authentication"
- "List all TypeScript components"
- "Find Rust structs and their methods"
- "Show me Go interfaces and implementations"
- "Find all C++ operator overloads in the Matrix class"
- "Show me C++ template functions with their specializations"
- "List all C++ namespaces and their contained classes"
- "Find C++ lambda expressions used in algorithms"
- "Show me all Solidity smart contracts and their events"
- "Find all modifier functions in the contract"
- "What state variables does this contract use?"
- "List all AutoHotkey hotkeys and their modifiers"
- "Find hotstring definitions and their replacements"
- "Show me all labels in the AHK script"
- "Add logging to all database connection functions"
- "Refactor the User class to use dependency injection"
- "Convert these Python functions to async/await pattern"
- "Add error handling to authentication methods"
- "Optimize this function for better performance"

### Automatic Parallel Execution (No Explicit Request Needed!)
For eligible read-only queries such as multi-file search, audits, and scoped analysis, the system automatically:
1. Shows a green notification: `✅ Auto-activating parallel execution: [task_type] (confidence: 0.xx)`
2. Previews a scope-aware file split and only keeps safe subtasks inside the requested scope
3. Distributes those subtasks across the configured worker pool, scaled down to the safe subtask count and CPU guardrails
4. Aggregates evidence-based worker output, including execution metadata and unresolved conflicts, into the final answer
5. Shows execution time: `⚡ Parallel execution completed in [X]s`

If you want to disable automatic parallelism for a specific run, use the `--no-parallel` flag:
```bash
cgr start --repo-path /path/to/repo --no-parallel
```

You can also inspect the execution plan without running worker LLM calls:
```bash
cgr start --parallel-dry-run --parallel-workers 8 --auto-split
```

### Step 3: Export Graph Data

For programmatic access and integration with other tools, you can export the entire knowledge graph to JSON:

**Export during graph update:**
```bash
cgr start --repo-path /path/to/repo --index-code --clean -o my_graph.json
```

**Export existing graph without updating:**
```bash
cgr export -o my_graph.json
```

**Optional: adjust Memgraph batching during export:**
```bash
cgr export -o my_graph.json --batch-size 5000
```

**Working with exported data:**
```python
from codebase_rag.graph_loader import load_graph

# Load the exported graph
graph = load_graph("my_graph.json")

# Get summary statistics
summary = graph.summary()
print(f"Total nodes: {summary['total_nodes']}")
print(f"Total relationships: {summary['total_relationships']}")

# Find specific node types
functions = graph.find_nodes_by_label("Function")
classes = graph.find_nodes_by_label("Class")

# Analyze relationships
for func in functions[:5]:
    relationships = graph.get_relationships_for_node(func.node_id)
    print(f"Function {func.properties['name']} has {len(relationships)} relationships")
```

**Example analysis script:**
```bash
python examples/graph_export_example.py my_graph.json
```

This provides a reliable, programmatic way to access your codebase structure without LLM restrictions, perfect for:
- Integration with other tools
- Custom analysis scripts
- Building documentation generators
- Creating code metrics dashboards

### JSON Data Ingestion (New!)
Import custom domain knowledge, entities, relationships, and metadata directly into your knowledge graph using structured JSON files. This feature lets you extend the auto-parsed code graph with custom business logic, external metadata, domain-specific entities, and custom relationships.

---

#### 1. Core Concepts & Key Features (MECE-Compliant, No Overlap)
| Category | Features |
|----------|----------|
| **Entity Management** | Import custom entities with any schema/label; reference auto-parsed code entities via qualified names |
| **Relationship Management** | Add custom relationships between any nodes (custom ↔ custom, custom ↔ auto-parsed) |
| **Dataset Isolation** | All imported data is grouped by `dataset_id` for independent management, updates, and deletion |
| **Ingest Control** | Dry run previews, batch processing of multiple files, idempotent operation, configurable conflict resolution, automatic ingestion during `cgr start` document indexing |
| **Parallel Processing** | Up to 32 parallel workers, FIFO/round-robin scheduling, per-file locking for deterministic parallel ingestion with no duplicate processing |
| **Vector Integration** | Embeddings automatically generated for all text properties for semantic search |
| **API Access** | First-class Python API + CLI support for programmatic and interactive usage |

*All features are mutually exclusive with no overlapping functionality, and collectively cover all custom ingestion use cases.*

---

#### 2. JSON Schema Specification (Deterministic, Fully Validated)
All JSON input is strictly validated against [ingestion_schema.json](./ingestion_schema.json) for deterministic behavior. NO TRANSFORMATIONS OR CONVERSIONS ARE PERFORMED DURING INGESTION - the schema is the single source of truth.

For schema definition, see [ingestion_schema.json](./ingestion_schema.json).
      "properties": "object (optional, key-value properties for the relationship)"
    }
  ]
}
```
**Validation Guarantees:**
- Missing required fields fail fast with clear error messages
- ID uniqueness is enforced per dataset
- Relationship IDs must exist either in the current import or the existing graph
- Property values are type-checked for consistency

---

#### 3. Programmatic Usage (Python API, Separation of Concerns)
Use the Python API for scripted integration and workflows:
```python
from codebase_rag.json_ingest import ingest_json_data, delete_dataset

# Ingest JSON data programmatically
result = ingest_json_data(
    repo_path="/path/to/repo",
    input_path="./custom_data.json",
    dataset_id="my_dataset",
    dry_run=False,
    conflict_strategy="overwrite"  # Options: overwrite, skip, fail
)
# Result object contains deterministic counts:
# - nodes_created, nodes_updated, nodes_skipped
# - relationships_created, relationships_skipped
# - errors (list of validation or processing errors)

# Delete all data in a dataset (atomic operation)
delete_result = delete_dataset(repo_path="/path/to/repo", dataset_id="my_dataset")
print(f"Deleted {delete_result['nodes_deleted']} nodes, {delete_result['relationships_deleted']} relationships")
```
*API maintains strict separation of concerns: ingestion logic is isolated from graph storage and embedding generation.*

---

#### 4. Reliability Guarantees (Deterministic & Correct)
The JSON ingestion system provides these deterministic guarantees:
- **Atomicity:** Ingest operations are all-or-nothing - partial failures leave the graph unchanged
- **Idempotency:** Re-running the same import with the same data will not create duplicate nodes/relationships
- **Consistency:** All nodes, relationships, and embeddings are created transactionally
- **Dry Run Accuracy:** Dry run mode produces exactly the same change summary as a real run, with no modifications
- **Error Determinism:** Same input will always produce the same error or success result
- **Conflict Resolution Predictability:** Configurable strategies (`overwrite`, `skip`, `fail`) produce consistent results for duplicate entities

---

#### 5. CLI Command Usage
See [JSON Data Ingestion Commands](#json-data-ingestion-commands) in the CLI section for interactive usage examples.

### Step 4: Document GraphRAG (New!)

Index and query documentation alongside your codebase. This feature enables comprehensive RAG across both code and documentation, with powerful validation capabilities.

#### Unified Dual-Graph Session

The `cgr start` command now supports unified dual-graph querying from a single session:

```bash
# Index both code and documents, then query both graphs
cgr start --repo-path /path/to/your/repo --index-all --with-docs --mode both_merged

# Index both code and documents + automatically ingest valid JSON files from repo root (10 parallel workers by default)
cgr start --repo-path /path/to/your/repo --index-all --with-docs --mode both_merged --ingest-json

# Index docs + ingest specific JSON file with 20 parallel workers and round-robin scheduling
cgr start --repo-path /path/to/your/repo --index-docs --with-docs --ingest-json \
  --json-path ./custom_domain_data.json \
  --json-workers 20 \
  --scheduling-strategy round-robin

# Index docs + ingest JSON, fail immediately if any file is invalid
cgr start --repo-path /path/to/your/repo --index-docs --with-docs --ingest-json \
  --json-fail-on-invalid

# Connect to document graph for specification validation
cgr start --repo-path /path/to/your/repo --with-docs --mode code_vs_doc

# Index documents only, then query document graph
cgr start --repo-path /path/to/your/repo --index-docs --with-docs --mode document_only
```

**Unified start flags:**

| Flag | Description |
|------|-------------|
| `--with-docs` | Connect to document graph for dual-graph querying |
| `--index-docs` | Index documents before starting chat |
| `--index-all` | Index both code and documents before starting chat |
| `--mode` | Query routing mode (see below) |
| `--doc-workspace` | Document workspace identifier (default: `default`) |
| `--check-freshness/--no-check-freshness` | Check if indexed graphs are up-to-date and prompt for reindex if stale (default: enabled) |
| `--index-timeout` | Maximum time in seconds for indexing operations (default: 300) |
| `--realtime-updater` | **NEW**: Enable real-time file system monitoring and automatic graph updates |
| `--realtime-debounce` | **NEW**: Debounce delay in seconds for realtime updates (default: `5.0`) |
| `--realtime-max-wait` | **NEW**: Maximum wait time in seconds before processing changes (default: `30.0`) |
| `--realtime-code` | **NEW**: Enable realtime updates for code files (default: `true`) |
| `--realtime-docs` | **NEW**: Enable realtime updates for document files (default: `false`) |
| `--realtime-json` | **NEW**: Enable realtime updates for JSON files (default: `false`) |
| `--ingest-json` | **NEW**: Enable automatic JSON ingestion during document indexing (validates against [ingestion_schema.json](./ingestion_schema.json)) |
| `--json-path` | **NEW**: Path to specific JSON file or directory to ingest (defaults to scanning repo root for all *.json files if not provided) |
| `--json-skip-invalid/--json-fail-on-invalid` | **NEW**: Skip invalid JSON files (default) or fail ingestion if any JSON file fails schema validation |
| `--json-workers` | **NEW**: Number of parallel workers for JSON ingestion (default: 10, max: 32) |
| `--scheduling-strategy` | **NEW**: Scheduling strategy for parallel JSON ingestion: `fifo` (default) or `round-robin` (even distribution of large/small files across workers) |

**In-chat mode switching:**

Use `/mode <mode>` to switch query modes during the session:

```
/mode both_merged     # Query both code and documents
/mode code_vs_doc     # Validate code against specs
/mode document_only   # Query documents only
```

#### Separate Document Commands

**Index documents into the document graph:**
```bash
cgr index-docs --repo-path /path/to/your/repo
```

This indexes Markdown, PDF, DOCX, and text files, creating Document, Section, and Chunk nodes with embeddings.

**For the first document indexing (clean start):**
```bash
cgr index-docs --repo-path /path/to/your/repo --clean
```

The `--clean` flag clears the document database before indexing, similar to the code GraphRAG `--clean` option.

**Query the document graph only:**
```bash
cgr query-docs "How do I use the authentication API?" --repo-path /path/to/your/repo
```

**Query both code and document graphs (merged results):**
```bash
cgr query-all "Tell me everything about authentication" --repo-path /path/to/your/repo
```

**Validate code against specification documents:**
```bash
# Check if code implements all endpoints in OpenAPI spec
cgr validate-spec \
  --repo-path /path/to/your/repo \
  --spec-path docs/api-spec.md \
  --scope all
```

**Validate documentation against actual code:**
```bash
# Check if documentation is still accurate
cgr validate-doc \
  --repo-path /path/to/your/repo \
  --doc-path docs/api.md \
  --scope sections
```

**Validation options:**
- `--scope`: Validation scope (`all`, `sections`, or `claims`)
- `--max-cost`: Maximum cost budget in USD (default: 0.50)
- `--dry-run`: Estimate cost without running validation

**Query Modes:**
- `code_only`: Query code graph only (for function lookups, call graphs)
- `document_only`: Query document graph only (for tutorials, guides, API docs)
- `both_merged`: Query both graphs, merge results with clear attribution
- `code_vs_doc`: Validate code against document specifications (doc is truth)
- `doc_vs_code`: Validate docs against actual code (code is truth)

**Use Cases:**
- **API Documentation Compliance**: Ensure code implements all specified endpoints
- **Documentation Audits**: Find outdated or incorrect documentation
- **Comprehensive Search**: Get answers spanning both code and documentation

<a id="json-data-ingestion-commands"></a>
#### JSON Data Ingestion Commands (Logical, Deterministic)
All CLI commands produce deterministic, consistent output and follow the same reliability guarantees as the Python API.

> **💡 Automatic Ingestion Tip**: You can also run JSON ingestion automatically during document indexing by adding the `--ingest-json` flag to `cgr start` (see [Unified start flags](#unified-start-flags)). This eliminates the need to run a separate `cgr ingest-json` command after indexing docs.

##### 🔹 `ingest-json` - Import custom JSON data
| Flag | Description | Behavior |
|------|-------------|----------|
| `INPUT_PATH` (positional) | Path to JSON file or directory of JSON files | Deterministic processing order: files sorted alphanumerically |
| `--dataset-id <id>` | Unique dataset identifier for isolation | All imported data is tagged with this ID for independent management |
| `--dry-run` | Preview changes without modifying graph | Outputs exact counts of nodes/relationships that would be created/updated |
| `--conflict-resolution <strategy>` | Handling for duplicate entity IDs | Options: `last-write-wins` (default), `highest-confidence-wins`, `manual-review` |
| `--batch-size <n>` | Number of operations per transaction | Default: 100, adjust for large imports |
| `--skip-existing` | Skip entities/relationships that already exist | Useful for incremental updates |
| `--incremental` | Run incremental update, only process changed entities/relationships | Optimizes performance for large datasets |

**Examples:**
```bash
# Ingest a single file with default settings
cgr ingest-json my_data.json

# Ingest directory with explicit conflict strategy
cgr ingest-json ./custom_data/ --conflict-resolution highest-confidence-wins

# Dry run to preview changes before applying
cgr ingest-json my_data.json --dry-run

# Import with custom dataset ID
cgr ingest-json my_data.json --dataset-id business_rules

# Skip existing entities and relationships
cgr ingest-json my_data.json --skip-existing

# Incremental update for large datasets
cgr ingest-json my_data.json --incremental --batch-size 500
```

##### 🔹 `delete-dataset` - Delete all data in a dataset
Atomic operation that removes all nodes and relationships belonging to a specific dataset, with no impact on other data.

```bash
# Delete all data in the "business_rules" dataset
cgr delete-dataset business_rules

# Dry run to see what would be deleted
cgr delete-dataset business_rules --dry-run
```
*Operation is idempotent: deleting a non-existent dataset returns a success with 0 items deleted.*

### YOLO Mode

YOLO mode disables all interactive confirmations for tool operations, automatically approving file edits, shell commands, and other potentially destructive operations without prompting.

**When to use:**
- Automated/scripted workflows
- CI/CD pipelines
- Non-interactive sessions
- When you trust the agent completely

**Usage:**
```bash
# Start with yolo mode (no confirmations)
cgr start --repo-path /path/to/repo --yolo

# Short form
cgr start -r /path/to/repo -y

# Backward compatible alias
cgr start --repo-path /path/to/repo --no-confirm

# For optimization sessions
cgr optimize python --repo-path /path/to/repo --yolo

# MCP server with yolo mode (via environment variable)
CGR_YOLO_MODE=true cgr mcp-server
```

**Environment Variable:**
- `CGR_YOLO_MODE=true` - Enable yolo mode for MCP server or persistent settings

**Precedence Rules:**
1. CLI flags (`--yolo`, `--no-confirm`) take precedence over environment variable
2. Environment variable (`CGR_YOLO_MODE=true`) takes precedence over default
3. Default: yolo mode disabled, confirmations enabled

**Security Warning:**
When yolo mode is enabled, a prominent red warning banner is displayed at session start. All auto-approved actions are logged with `YOLO:` prefix for audit trail. Use with caution on production codebases.

## 🔒 Security Best Practices
Follow these recommendations to ensure secure deployment and usage of Code-Graph-RAG:

### Database Security
- **Always enable authentication** for Memgraph instances, especially when deployed in shared environments or exposed to networks
- **Bind database services to `localhost` only** by default. Never use `0.0.0.0` as the host binding unless you explicitly intend to expose the service to external networks
- Use strong, unique passwords for all database accounts
- For production deployments, use TLS encryption for all database connections

### Submodule & Supply Chain Security
- All official Tree-sitter grammar submodules are pinned to specific, audited commit hashes to prevent supply chain attacks from malicious upstream changes
- When adding custom Tree-sitter grammars, always pin them to specific commit hashes and verify the source before use
- Regularly update dependencies to patch security vulnerabilities

### Document Parsing Security
- File operations include path traversal protection by default. When global filesystem access is disabled, all operations are restricted to the target repository directory. When enabled, path validation still prevents traversal attacks while allowing access to other locations.
- Symlinks are validated to ensure they point to allowed locations
- File size limits prevent denial-of-service attacks from oversized files
- Untrusted file formats (PDF, DOCX) are parsed in isolated environments when enabled

### Global Filesystem Access Security
- Global filesystem access is enabled by default to allow flexible file operations across your system
- Write operations outside the project repository require explicit approval by default, to prevent accidental or malicious modifications
- Disable global access in shared or untrusted environments by setting `ENABLE_GLOBAL_FILE_ACCESS=false` in your `.env` file
- All file operations are logged for audit purposes, including access to locations outside the project repository

### Network Security
- The MCP server binds to `127.0.0.1` by default to prevent external access
- All external API calls (to LLM providers) use HTTPS encryption
- Never expose the MCP server or database instances directly to the public internet without proper authentication and access controls

### Yolo Mode Security
- Yolo mode is intended for testing, CI/CD pipelines, and trusted environments only
- Never enable Yolo mode for untrusted workloads or production codebases unless you fully understand the risks
- All Yolo mode actions are logged for audit purposes

## 🔌 MCP Server (Claude Code Integration)

Code-Graph-RAG can run as an MCP (Model Context Protocol) server, enabling seamless integration with Claude Code and other MCP clients.

### Quick Setup

```bash
claude mcp add --transport stdio code-graph-rag \
  --env TARGET_REPO_PATH=/absolute/path/to/your/project \
  --env CYPHER_PROVIDER=openai \
  --env CYPHER_MODEL=gpt-4 \
  --env CYPHER_API_KEY=your-api-key \
  -- uv run --directory /path/to/code-graph-rag code-graph-rag mcp-server
```

### Available Tools

<!-- SECTION:mcp_tools -->
| Tool | Description |
|----|-----------|
| `list_projects` | List all indexed projects in the knowledge graph database. Returns a list of project names that have been indexed. |
| `delete_project` | Delete a specific project from the knowledge graph database. This removes all nodes associated with the project while preserving other projects. Use list_projects first to see available projects. |
| `wipe_database` | WARNING: Completely wipe the entire database, removing ALL indexed projects. This cannot be undone. Use delete_project for removing individual projects. |
| `index_repository` | WARNING: Clears all data for the current project including its embeddings. Parse and ingest the repository into the Memgraph knowledge graph. Use update_repository for incremental updates. Only use when explicitly requested. |
| `update_repository` | Update the repository in the Memgraph knowledge graph without clearing existing data. Use this for incremental updates. |
| `query_code_graph` | Query the codebase knowledge graph using natural language. Use semantic_search unless you know the exact names of classes/functions you are searching for. Ask questions like 'What functions call UserService.create_user?' or 'Show me all classes that implement the Repository interface'. |
| `get_code_snippet` | Retrieve source code for a function, class, or method by its qualified name. Returns the source code, file path, line numbers, and docstring. |
| `surgical_replace_code` | Surgically replace an exact code block in a file using diff-match-patch. Only modifies the exact target block, leaving the rest unchanged. |
| `read_file` | Read the contents of a file from the project. Supports pagination for large files. |
| `write_file` | Write content to a file, creating it if it doesn't exist. |
| `list_directory` | List contents of a directory in the project. |
| `semantic_search` | Performs a semantic search for functions based on a natural language query describing their purpose, returning a list of potential matches with similarity scores. Requires the 'semantic' extra to be installed. |
| `ask_agent` | Ask the Code Graph RAG agent a question about the codebase. Uses the full RAG pipeline to analyze the code graph and provide a detailed answer. Use this for general questions about architecture, functionality, and code relationships. |
| `get_embedding_status` | Get the current embedding provider configuration and status. Returns the current provider, model, dimension, and available providers. |
| `set_embedding_provider` | Switch to a different embedding provider. Supported providers: local, openai, google, ollama. Optionally re-embed all vectors after switching. |
| `query_document_graph` | **📚 NEW**: Query the DOCUMENT graph/vector ONLY. Use for questions about documentation, specifications, and textual content. Returns relevant document sections and chunks. |
| `query_both_graphs` | **📚 NEW**: Query BOTH code and document graphs, merge results. Use for comprehensive searches spanning code and documentation. Results are labeled with their source (code_graph or document_graph). |
| `validate_code_against_spec` | **📚 NEW**: Validate CODE against DOCUMENT specifications. Checks if the implementation matches the specification document. Returns validation report with discrepancies. |
| `validate_doc_against_code` | **📚 NEW**: Validate DOCUMENT against actual CODE. Checks if documentation accurately reflects the current code. Identifies outdated or incorrect documentation. |
| `index_documents` | **📚 NEW**: Index documents into the document graph. Parses and ingests markdown, PDF, DOCX, and text files. Creates Document, Section, and Chunk nodes with embeddings. |
<!-- /SECTION:mcp_tools -->

### Example Usage

```
> Index this repository
> What functions call UserService.create_user?
> Update the login function to add rate limiting
```

For detailed setup, see [Claude Code Setup Guide](docs/claude-code-setup.md).

## 📊 Graph Schema

The knowledge graph uses the following node types and relationships:

### Node Types

<!-- SECTION:node_schemas -->
| Label | Properties |
|-----|----------|
| Project | `{name: string}` |
| Package | `{qualified_name: string, name: string, path: string, absolute_path: string}` |
| Folder | `{path: string, name: string, absolute_path: string}` |
| File | `{path: string, name: string, extension: string, absolute_path: string}` |
| Module | `{qualified_name: string, name: string, path: string, absolute_path: string}` |
| Class | `{qualified_name: string, name: string, decorators: list[string], path: string, absolute_path: string}` |
| Function | `{qualified_name: string, name: string, decorators: list[string], path: string, absolute_path: string}` |
| Method | `{qualified_name: string, name: string, decorators: list[string], path: string, absolute_path: string}` |
| Interface | `{qualified_name: string, name: string, path: string, absolute_path: string}` |
| Enum | `{qualified_name: string, name: string, path: string, absolute_path: string}` |
| Type | `{qualified_name: string, name: string}` |
| Union | `{qualified_name: string, name: string}` |
| ModuleInterface | `{qualified_name: string, name: string, path: string, absolute_path: string}` |
| ModuleImplementation | `{qualified_name: string, name: string, path: string, absolute_path: string, implements_module: string}` |
| ExternalPackage | `{name: string, version_spec: string}` |
| Contract | `{qualified_name: string, name: string, is_abstract: bool, path: string, absolute_path: string, start_line: int, end_line: int}` |
| Library | `{qualified_name: string, name: string, path: string, absolute_path: string, start_line: int, end_line: int}` |
| Event | `{qualified_name: string, name: string, parameters: list[string], is_anonymous: bool, indexed_count: int, path: string, absolute_path: string, start_line: int, end_line: int}` |
| Modifier | `{qualified_name: string, name: string, parameters: list[string], path: string, absolute_path: string, start_line: int, end_line: int}` |
| StateVariable | `{qualified_name: string, name: string, type: string, visibility: string, is_constant: bool, is_immutable: bool, is_mapped: bool, path: string, absolute_path: string, start_line: int, end_line: int}` |
| CustomError | `{qualified_name: string, name: string, parameters: list[string], path: string, absolute_path: string, start_line: int, end_line: int}` |
| **Document** 📚 | `{qualified_name: string, path: string, file_type: string, word_count: int, total_section_count: int, modified_date: string, workspace: string}` |
| **Section** 📚 | `{qualified_name: string, title: string, level: int, start_line: int, end_line: int, content_snippet: string, workspace: string}` |
| **Chunk** 📚 | `{qualified_name: string, content: string, start_line: int, end_line: int, embedding: vector, workspace: string}` |
<!-- /SECTION:node_schemas -->

### Language-Specific Mappings

<!-- SECTION:language_mappings -->
- **AutoHotkey**: `class_definition`, `function_definition`, `hotkey`, `hotstring_definition`, `label`
- **C**: `enum_specifier`, `function_definition`, `struct_specifier`, `union_specifier`
- **C++**: `class_specifier`, `declaration`, `enum_specifier`, `field_declaration`, `function_definition`, `lambda_expression`, `struct_specifier`, `template_declaration`, `union_specifier`
- **Java**: `annotation_type_declaration`, `class_declaration`, `constructor_declaration`, `enum_declaration`, `interface_declaration`, `method_declaration`, `record_declaration`
- **JavaScript**: `arrow_function`, `class`, `class_declaration`, `function_declaration`, `function_expression`, `generator_function_declaration`, `method_definition`
- **Lua**: `function_declaration`, `function_definition`
- **PHP**: `anonymous_function`, `arrow_function`, `class_declaration`, `enum_declaration`, `function_definition`, `interface_declaration`, `method_declaration`, `trait_declaration`
- **Python**: `class_definition`, `function_definition`
- **Rust**: `closure_expression`, `enum_item`, `function_item`, `function_signature_item`, `impl_item`, `struct_item`, `trait_item`, `type_item`, `union_item`
- **Solidity**: `constructor_definition`, `contract_declaration`, `enum_declaration`, `fallback_receive_definition`, `function_definition`, `interface_declaration`, `library_declaration`, `modifier_definition`, `struct_declaration`
- **TypeScript**: `abstract_class_declaration`, `arrow_function`, `class`, `class_declaration`, `enum_declaration`, `function_declaration`, `function_expression`, `function_signature`, `generator_function_declaration`, `interface_declaration`, `internal_module`, `method_definition`, `type_alias_declaration`
- **C#**: `anonymous_method_expression`, `class_declaration`, `constructor_declaration`, `destructor_declaration`, `enum_declaration`, `function_pointer_type`, `interface_declaration`, `lambda_expression`, `local_function_statement`, `method_declaration`, `struct_declaration`
- **Go**: `function_declaration`, `method_declaration`, `type_declaration`
- **Scala**: `class_definition`, `function_declaration`, `function_definition`, `object_definition`, `trait_definition`
<!-- /SECTION:language_mappings -->

### Relationships

<!-- SECTION:relationship_schemas -->
| Source | Relationship | Target |
|------|------------|------|
| Project, Package, Folder | CONTAINS_PACKAGE | Package |
| Project, Package, Folder | CONTAINS_FOLDER | Folder |
| Project, Package, Folder | CONTAINS_FILE | File |
| Project, Package, Folder | CONTAINS_MODULE | Module |
| Module | DEFINES | Class, Function, Interface, Enum, Type, Union, Contract, Library |
| Class, Contract | DEFINES_METHOD | Method |
| Module | IMPORTS | Module |
| Module | EXPORTS | Class, Function, Interface, Enum, Type, Union |
| Module | EXPORTS_MODULE | ModuleInterface |
| Module | IMPLEMENTS_MODULE | ModuleImplementation |
| Class, Contract | INHERITS | Class, Contract |
| Class, Contract | IMPLEMENTS | Interface |
| Method | OVERRIDES | Method |
| ModuleImplementation | IMPLEMENTS | ModuleInterface |
| Project | DEPENDS_ON_EXTERNAL | ExternalPackage |
| Function, Method | CALLS | Function, Method |
| Function, Method | EMITS | Event |
| Function, Method | MODIFIED_BY | Modifier |
| Contract, Library | USES_LIBRARY | Library |
| Function, Method | CALLS_EXTERNAL | Function, Method |
| Contract, Interface | DEFINES_EVENT | Event |
| Contract | DEFINES_MODIFIER | Modifier |
| Contract | DEFINES_STATE | StateVariable |
| Function, Method | CALLS_DELEGATE | Function, Method |
| Function, Method | CALLS_STATIC | Function, Method |
| Function, Method | READS_STATE | StateVariable |
| Function, Method | WRITES_STATE | StateVariable |
| Function, Method | REVERTS_WITH | CustomError |
| **📚 Document** | CONTAINS_SECTION | **Section** |
| **📚 Section** | HAS_SUBSECTION | **Section** |
| **📚 Section** | HAS_CHUNK | **Chunk** |
| **📚 Chunk** | BELONGS_TO_SECTION | **Section** |
| **💎 Solidity Contract** | USES_LIBRARY | **Library** |
| **💎 Solidity Function** | EMITS | **Event** |
| **💎 Solidity Function** | MODIFIED_BY | **Modifier** |
| **💎 Solidity Function** | READS_STATE | **StateVariable** |
| **💎 Solidity Function** | WRITES_STATE | **StateVariable** |
| **💎 Solidity Function** | REVERTS_WITH | **CustomError** |
<!-- /SECTION:relationship_schemas -->

## 🔧 Configuration

Configuration is managed through environment variables in `.env` file:

### Custom Environment File Path
- `ENV_FILE`: Path to a custom environment file to load instead of the default `.env` (e.g. `ENV_FILE=.env_for_cgr`). This allows you to use separate configuration files for different repositories or environments.

When using a custom env file, you can either:
1. Run from the target repository root: `ENV_FILE=.env_for_cgr cgr start`
2. Run from a different directory: `ENV_FILE=/full/path/to/.env_for_cgr cgr start --repo-path /full/path/to/target/repo`

Values from the custom `ENV_FILE` will take precedence over any values in a default `.env` file in the current working directory.

### Provider-Specific Settings

#### Orchestrator Model Configuration
- `ORCHESTRATOR_PROVIDER`: Provider name (`google`, `openai`, `ollama`, `anthropic`)
- `ORCHESTRATOR_MODEL`: Model ID (e.g., `gemini-2.5-pro`, `gpt-4o`, `llama3.2`)
- `ORCHESTRATOR_API_KEY`: API key for the provider (if required)
- `ORCHESTRATOR_ENDPOINT`: Custom endpoint URL (if required)
- `ORCHESTRATOR_PROJECT_ID`: Google Cloud project ID (for Vertex AI)
- `ORCHESTRATOR_REGION`: Google Cloud region (default: `us-central1`)
- `ORCHESTRATOR_PROVIDER_TYPE`: Google provider type (`gla` or `vertex`)
- `ORCHESTRATOR_THINKING_BUDGET`: Thinking budget for reasoning models
- `ORCHESTRATOR_SERVICE_ACCOUNT_FILE`: Path to service account file (for Vertex AI)

#### Cypher Model Configuration
- `CYPHER_PROVIDER`: Provider name (`google`, `openai`, `ollama`, `anthropic`)
- `CYPHER_MODEL`: Model ID (e.g., `gemini-2.5-flash`, `gpt-4o-mini`, `codellama`)
- `CYPHER_API_KEY`: API key for the provider (if required)
- `CYPHER_ENDPOINT`: Custom endpoint URL (if required)
- `CYPHER_PROJECT_ID`: Google Cloud project ID (for Vertex AI)
- `CYPHER_REGION`: Google Cloud region (default: `us-central1`)
- `CYPHER_PROVIDER_TYPE`: Google provider type (`gla` or `vertex`)
- `CYPHER_THINKING_BUDGET`: Thinking budget for reasoning models
- `CYPHER_SERVICE_ACCOUNT_FILE`: Path to service account file (for Vertex AI)

#### Parallel Sub-Agent Worker LLM Configuration
Configure dedicated LLMs for parallel sub-agent workers to optimize cost/performance for parallel workloads:
- `CGR_WORKER_LLMS`: Comma-separated list of model strings (e.g., `openai:gpt-4o-mini,anthropic:claude-3-haiku`) or JSON array of full model configurations for sub-agent workers
- `CGR_WORKER_LLM_ASSIGNMENT_STRATEGY`: LLM assignment strategy for sub-agents. Only `round-robin` is supported currently, which evenly distributes configured LLMs across workers.

#### Automatic Parallel Execution Configuration
No explicit user request is required for safe read-only tasks. The runtime previews subtasks, blocks write-like prompts from parallel execution, and uses the configured worker pool only when at least two safe subtasks remain:
- `CGR_AUTO_PARALLEL_ENABLED`: Enable automatic parallel execution detection (default: `true`)
- `CGR_DEFAULT_PARALLEL_WORKERS`: Default number of parallel workers (default: `20`)
- `CGR_MAX_PARALLEL_WORKERS`: Maximum allowed parallel workers (default: `30`)
- `CGR_ALLOW_DYNAMIC_MAX_OVERRIDE`: Allow overriding the max worker limit for large workloads (default: `true`)
- `CGR_AUTO_SCALE_WORKERS`: Automatically scale worker count to match number of subtasks (default: `true`)
- `CGR_PARALLEL_ELIGIBILITY_THRESHOLD`: Confidence threshold for automatic parallel activation (default: `0.7`)
- `CGR_PARALLEL_MAX_QUEUE_SIZE`: Maximum size of the parallel subtask queue before runtime falls back to sequential behavior (default: `100`)
- `CGR_SUBAGENT_TIMEOUT`: Timeout per subtask execution in seconds (default: `300`)
- `CGR_SUBAGENT_ALLOW_WRITE`: Worker write access toggle (default: `false`; write-like tasks stay sequential in v1)
- `CGR_AUTO_SPLIT_ENABLED`: Enable automatic task splitting for parallel execution (default: `true`)
- `CGR_SUBAGENT_RETRY_ATTEMPTS`: Number of retry attempts for failed subtasks (default: `2`)

**CLI Overrides**: `--parallel-workers`, `--auto-split/--no-auto-split`, `--parallel-dry-run`, `--scheduling-strategy`, and `--no-parallel` all affect the interactive runtime path.

**Key Behavior**:
- If no worker LLMs are configured, sub-agents automatically use the orchestrator LLM as default
- The dedicated Cypher LLM remains unchanged and is shared by all workers for graph query generation
- All worker LLM configurations are validated before starting parallel execution, with clear error messages for missing API keys or invalid model identifiers

**Example Static Configuration**:
```bash
# Simple comma-separated config (reuses orchestrator provider credentials)
CGR_WORKER_LLMS="openai:Doubao-Seed-2.0-Code,openai:MiniMax-M2.5,openai:Kimi-K2.5,openai:GLM-4.7,openai:DeepSeek-V3.2,openai:Doubao-Seed-2.0-pro"

# Full JSON config with custom credentials/endpoints
CGR_WORKER_LLMS='[
  {"provider": "openai", "model_id": "Doubao-Seed-2.0-Code", "api_key": "your-api-key", "endpoint": "https://ark.cn-beijing.volces.com/api/coding/v3"},
  {"provider": "openai", "model_id": "MiniMax-M2.5", "api_key": "your-api-key", "endpoint": "https://ark.cn-beijing.volces.com/api/coding/v3"}
]'
```

**Dynamic Prompt Configuration**:
You can also specify worker LLMs directly in your natural language prompts without modifying config files:
```
"Review all Python files with 10 parallel workers using Doubao-Seed-2.0-Code and MiniMax-M2.5"
"Use Kimi-K2.5 and GLM-4.7 as worker LLMs for this parallel code generation task"
```
The system automatically detects the requested worker LLMs, validates them, and configures the parallel sub-agent pool accordingly.

#### Embedding Provider Configuration

Code-Graph-RAG supports multiple embedding providers for semantic search:

- `EMBEDDING_PROVIDER`: Provider name (`local`, `openai`, `google`, `ollama`) - default: `local`
- `EMBEDDING_MODEL`: Model identifier (e.g., `microsoft/unixcoder-base`, `text-embedding-3-small`, `nomic-embed-text`)
- `EMBEDDING_API_KEY`: API key for external providers (optional if set via provider-specific env vars)
- `EMBEDDING_ENDPOINT`: Custom endpoint or base URL for embedding APIs (optional)
- `EMBEDDING_DEVICE`: Device for local models (`auto`, `cpu`, `cuda`) - default: `auto`
- `EMBEDDING_KEEP_ALIVE`: Ollama model keep-alive duration (e.g., `5m`)
- `EMBEDDING_PROJECT_ID`: Google Cloud project ID (for Vertex AI)
- `EMBEDDING_REGION`: Google Cloud region (default: `us-central1`)
- `EMBEDDING_PROVIDER_TYPE`: Google provider type (`gla` or `vertex`)

**Available Providers:**

| Provider | Models | Dimension | API Key Required |
|----------|--------|-----------|------------------|
| `local` | `microsoft/unixcoder-base` (default) | 768 | No |
| `openai` | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-v4` | 1536, 3072, 1024 | Yes |
| `google` | `text-embedding-004`, `embedding-001` | 768 | Yes (GLA) or Service Account (Vertex) |
| `ollama` | `nomic-embed-text`, `mxbai-embed-large`, etc. | Varies | No |

**Example Configuration:**

```bash
# Use OpenAI embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-your-key

# Use Ollama local embeddings
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_KEEP_ALIVE=5m

# Use Google Vertex AI
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_PROVIDER_TYPE=vertex
EMBEDDING_PROJECT_ID=your-project-id
```

### System Settings
- `MEMGRAPH_HOST`: Memgraph hostname (default: `localhost`)
- `MEMGRAPH_PORT`: Memgraph port (default: `7687`)
- `MEMGRAPH_HTTP_PORT`: Memgraph HTTP port (default: `7444`)
- `LAB_PORT`: Memgraph Lab port (default: `3000`)
- `MEMGRAPH_BATCH_SIZE`: Batch size for Memgraph operations (default: `1000`)
- `TARGET_REPO_PATH`: Default repository path (default: `.`)
- `LOCAL_MODEL_ENDPOINT`: Fallback endpoint for Ollama (default: `http://localhost:11434/v1`)
- `CGR_YOLO_MODE`: Enable yolo mode globally (default: `false`)
- `ENABLE_GLOBAL_FILE_ACCESS`: Enable access to files outside the project repository (default: `true`)
- `GLOBAL_FILE_ACCESS_WRITE_REQUIRES_APPROVAL`: Require explicit approval for write operations outside the project repository (default: `true`)
- `PARALLEL_INDEXING_WORKERS`: Default number of parallel workers for codebase indexing (default: `20`). Auto-optimized at runtime to never exceed available CPU cores or number of changed files. Set to `1` to disable parallel indexing entirely and run sequentially.

### Logging Configuration
All logs are written to the terminal by default. You can control the log level via environment variable or CLI flag:

- `CGR_LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, default: `INFO`)

#### Global CLI Logging Flags (Applies to All Commands)
| Flag | Description |
|------|-------------|
| `--log-level LEVEL` | Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Custom Ignore Patterns

You can specify additional files or directories to exclude by creating a `.cgrignore` file in your repository root:

```
# Comments start with #
vendor
.custom_cache
my_build_output
docs/tree-sitter.txt
*.txt
*.pdf
*.egg-info/
```

- One pattern per line
- Lines starting with `#` are comments
- Blank lines are ignored
- Patterns support exact paths, directory prefixes, and glob-style matches such as `*.txt` and `*.egg-info/`
- Prefixing a pattern with `!` unignores matching paths
- Patterns from `.cgrignore` are merged with `--exclude` flags and auto-detected directories
- The same ignore rules are applied consistently to code indexing and document indexing

### Key Dependencies

<!-- SECTION:dependencies -->
- **loguru**: Python logging made (stupidly) simple
- **mcp**: Model Context Protocol SDK
- **pydantic-ai**: Agent Framework / shim to use Pydantic with LLMs
- **pydantic-settings**: Settings management using Pydantic
- **pymgclient**: Memgraph database adapter for Python language
- **python-dotenv**: Read key-value pairs from a .env file and set them as environment variables
- **tiktoken**: tiktoken is a fast BPE tokeniser for use with OpenAI's models
- **toml**: Python Library for Tom's Obvious, Minimal Language
- **tree-sitter-python**: Python grammar for tree-sitter
- **tree-sitter-solidity**: Solidity grammar for tree-sitter (smart contract support)
- **tree-sitter-autohotkey**: AutoHotkey grammar for tree-sitter (AHK v1 scripting)
- **tree-sitter**: Python bindings to the Tree-sitter parsing library
- **watchdog**: Filesystem events monitoring
- **typer**: Typer, build great CLIs. Easy to code. Based on Python type hints.
- **rich**: Render rich text, tables, progress bars, syntax highlighting, markdown and more to the terminal
- **prompt-toolkit**: Library for building powerful interactive command lines in Python
- **diff-match-patch**: Repackaging of Google's Diff Match and Patch libraries.
- **click**: Composable command line interface toolkit
- **protobuf**
- **defusedxml**: XML bomb protection for Python stdlib modules
- **huggingface-hub**: Client library to download and publish models, datasets and other repos on the huggingface.co hub
<!-- /SECTION:dependencies -->

## 🤖 Agentic Workflow & Tools

The agent is designed with a deliberate workflow to ensure it acts with context and precision, especially when modifying the file system.

### Core Tools

The agent has access to a suite of tools to understand and interact with the codebase:

<!-- SECTION:agentic_tools -->
| Tool | Description |
|----|-----------|
| `query_graph` | Query the codebase knowledge graph using natural language questions. Ask in plain English about classes, functions, methods, dependencies, or code structure. Examples: 'Find all functions that call each other', 'What classes are in the user module', 'Show me functions with the longest call chains'. |
| `read_file` | Reads the content of text-based files. For documents like PDFs or images, use the 'analyze_document' tool instead. |
| `create_file` | Creates a new file with content. IMPORTANT: Check file existence first! Overwrites completely WITHOUT showing diff. Use only for new files, not existing file modifications. |
| `replace_code` | Surgically replaces specific code blocks in files. Requires exact target code and replacement. Only modifies the specified block, leaving rest of file unchanged. True surgical patching. |
| `list_directory` | Lists the contents of a directory to explore the codebase. |
| `analyze_document` | **📚 Analyzes documents (PDFs, images)** to answer questions about their content using Google's multimodal AI. |
| `execute_shell` | Executes shell commands from allowlist. Read-only commands run without approval; write operations require user confirmation. IMPORTANT: Shell redirect operators (> >> < << 2>/dev/null 2>&1 etc.) are NOT supported because direct process execution cannot interpret shell syntax. |
| `semantic_search` | Performs a semantic search for functions based on a natural language query describing their purpose, returning a list of potential matches with similarity scores. |
| `get_function_source` | Retrieves the source code for a specific function or method using its internal node ID, typically obtained from a semantic search result. |
| `get_code_snippet` | Retrieves the source code for a specific function, class, or method using its full qualified name. |
| `query_document_graph` | **📚 Query the DOCUMENT graph only.** Use for questions about documentation, tutorials, guides, or API docs. Returns relevant document sections and chunks. |
| `query_both_graphs` | **📚 Query BOTH code and document graphs.** Merged results with source attribution. Use for comprehensive searches spanning code and documentation. |
| `validate_code_against_spec` | **📚 Validate CODE against DOCUMENT specifications.** Checks if implementation matches spec documents. Returns validation report with discrepancies. |
| `validate_doc_against_code` | **📚 Validate DOCUMENT against actual CODE.** Identifies outdated or incorrect documentation. Returns validation report with suggestions. |
| `index_documents` | **📚 Index documents into the document graph.** Parses and ingests Markdown, PDF, DOCX files. Creates Document, Section, and Chunk nodes with embeddings. |
<!-- /SECTION:agentic_tools -->

### 📚 Document GraphRAG Tools

**Document Analysis**: The `analyze_document` tool uses Google's multimodal AI to analyze PDFs, images, and other documents, answering questions about their content directly.

**Document Graph Indexing**: Index structured documents (Markdown, PDF, DOCX) into the document graph with:
- **Document nodes**: Metadata including file type, word count, section count
- **Section nodes**: Hierarchical document structure with titles, levels, line numbers
- **Chunk nodes**: Text chunks with embeddings for semantic search

**Query Modes**:
- **code_only**: Query code graph only
- **document_only**: Query document graph only
- **both_merged**: Query both graphs with merged results
- **code_vs_doc**: Validate code against specifications
- **doc_vs_code**: Validate documentation against code

### Intelligent and Safe File Editing

The agent uses AST-based function targeting with Tree-sitter for precise code modifications. Features include:
- **Visual diff preview** before changes
- **Surgical patching** that only modifies target code blocks
- **Multi-language support** across all supported languages
- **Security sandbox** preventing edits outside project directory
- **Smart function matching** with qualified names and line numbers



## 🌍 Multi-Language Support

### Adding New Languages

Code-Graph-RAG makes it easy to add support for any language that has a Tree-sitter grammar. The system automatically handles grammar compilation and integration.

> **⚠️ Recommendation**: While you can add languages yourself, we recommend waiting for official full support to ensure optimal parsing quality, comprehensive feature coverage, and robust integration. The languages marked as "In Development" above will receive dedicated optimization and testing.

> **💡 Request Support**: If you want a specific language to be officially supported, please [submit an issue](https://github.com/vitali87/code-graph-rag/issues) with your language request.

#### Quick Start: Add a Language

Use the built-in language management tool to add any Tree-sitter supported language:

```bash
# Add a language using the standard tree-sitter repository
cgr language add-grammar <language-name>

# Examples:
cgr language add-grammar c-sharp
cgr language add-grammar php
cgr language add-grammar ruby
cgr language add-grammar kotlin
```

#### Custom Grammar Repositories

For languages hosted outside the standard tree-sitter organization:

```bash
# Add a language from a custom repository
cgr language add-grammar --grammar-url https://github.com/custom/tree-sitter-mylang
```

#### Recommended Grammar URLs

For languages not in the standard tree-sitter organization, use these verified grammars:

| Language | Grammar URL | Notes |
|----------|-------------|-------|
| **AutoHotkey** | `https://github.com/alfredomtx/tree-sitter-autohotkey` | AHK v1 scripting, includes LSP/debugger, hotkeys, hotstrings |
| **Solidity** | `https://github.com/JoranHonig/tree-sitter-solidity` | Most popular Solidity grammar (185+ stars), smart contracts |

**Example:**
```bash
# Add AutoHotkey support (Windows automation scripting)
cgr language add-grammar --grammar-url https://github.com/alfredomtx/tree-sitter-autohotkey

# Add Solidity support (Ethereum smart contracts)
cgr language add-grammar --grammar-url https://github.com/JoranHonig/tree-sitter-solidity
```

#### What Happens Automatically

When you add a language, the tool automatically:

1. **Downloads the Grammar**: Clones the tree-sitter grammar repository as a git submodule
2. **Detects Configuration**: Auto-extracts language metadata from `tree-sitter.json`
3. **Analyzes Node Types**: Automatically identifies AST node types for:
   - Functions/methods (`method_declaration`, `function_definition`, etc.)
   - Classes/structs (`class_declaration`, `struct_declaration`, etc.)
   - Modules/files (`compilation_unit`, `source_file`, etc.)
   - Function calls (`call_expression`, `method_invocation`, etc.)
   - **Language-specific features**: Hotkeys (AutoHotkey), events/modifiers (Solidity), etc.
4. **Compiles Bindings**: Builds Python bindings from the grammar source
5. **Updates Configuration**: Adds the language to `codebase_rag/language_config.py`
6. **Enables Parsing**: Makes the language immediately available for codebase analysis

#### Example: Adding C# Support

```bash
$ cgr language add-grammar c-sharp
🔍 Using default tree-sitter URL: https://github.com/tree-sitter/tree-sitter-c-sharp
🔄 Adding submodule from https://github.com/tree-sitter/tree-sitter-c-sharp...
✅ Successfully added submodule at grammars/tree-sitter-c-sharp
Auto-detected language: c-sharp
Auto-detected file extensions: ['cs']
Auto-detected node types:
Functions: ['destructor_declaration', 'method_declaration', 'constructor_declaration']
Classes: ['struct_declaration', 'enum_declaration', 'interface_declaration', 'class_declaration']
Modules: ['compilation_unit', 'file_scoped_namespace_declaration', 'namespace_declaration']
Calls: ['invocation_expression']

✅ Language 'c-sharp' has been added to the configuration!
📝 Updated codebase_rag/language_config.py
```

#### Managing Languages

```bash
# List all configured languages
cgr language list-languages

# Remove a language (this also removes the git submodule unless --keep-submodule is specified)
cgr language remove-language <language-name>
```

#### Language Configuration

The system uses a configuration-driven approach for language support. Each language is defined in `codebase_rag/language_config.py` with the following structure:

```python
"language-name": LanguageConfig(
    name="language-name",
    file_extensions=[".ext1", ".ext2"],
    function_node_types=["function_declaration", "method_declaration"],
    class_node_types=["class_declaration", "struct_declaration"],
    module_node_types=["compilation_unit", "source_file"],
    call_node_types=["call_expression", "method_invocation"],
),
```

#### Troubleshooting

**Grammar not found**: If the automatic URL doesn't work, use a custom URL:
```bash
cgr language add-grammar --grammar-url https://github.com/custom/tree-sitter-mylang
```

**Version incompatibility**: If you get "Incompatible Language version" errors, update your tree-sitter package:
```bash
uv add tree-sitter@latest
```

**Missing node types**: The tool automatically detects common node patterns, but you can manually adjust the configuration in `language_config.py` if needed.

## 📦 Building a binary

You can build a binary of the application using the `build_binary.py` script. This script uses PyInstaller to package the application and its dependencies into a single executable.

```bash
python build_binary.py
```
The resulting binary will be located in the `dist` directory.

## 🐛 Debugging

All logs are written to terminal by default. Use `--log-level DEBUG` flag for detailed troubleshooting output.
2. **Check Memgraph connection**:
   - Ensure Docker containers are running: `docker-compose ps`
   - Verify Memgraph is accessible on port 7687

2. **View database in Memgraph Lab**:
   - Open http://localhost:3000
   - Connect to memgraph:7687

3. **For local models**:
   - Verify Ollama is running: `ollama list`
   - Check if models are downloaded: `ollama pull llama3`
   - Test Ollama API: `curl http://localhost:11434/v1/models`
   - Check Ollama logs: `ollama logs`

## 🤝 Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

Good first PRs are from TODO issues.

## 🙋‍♂️ Support

For issues or questions:
1. Check the logs for error details
2. Verify Memgraph connection
3. Ensure all environment variables are set
4. Review the graph schema matches your expectations

## 💼 Enterprise Services

Code-Graph-RAG is open source and free to use. For organizations that need more, we offer **fully managed cloud-hosted solutions** and **on-premise deployments**:

- **Cloud-Hosted Deployment** — Managed cloud infrastructure for both the graph database and AI agent connection. Zero infrastructure overhead — we handle scaling, updates, and availability so your team can focus on building.
- **On-Premise & Air-Gapped Deployment** — Deploy Code-Graph-RAG entirely within your own environment, including air-gapped networks. Full data sovereignty for regulated industries and security-sensitive organizations.

We also offer custom development, integration consulting, technical support contracts, and team training.

**[View plans & pricing at code-graph-rag.com →](https://code-graph-rag.com/enterprise)**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vitali87/code-graph-rag&type=Date)](https://www.star-history.com/#vitali87/code-graph-rag&Date)

## Fork History

[![Fork History Chart](https://fork-history.site/svg?repos=vitali87/code-graph-rag)](https://fork-history.site/#vitali87/code-graph-rag)
