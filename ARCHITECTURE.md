# Code-Graph-RAG Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Code-Graph-RAG System                               │
│                         (Graph-Based RAG for Codebases)                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                   USER INTERFACE                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   CLI (cgr)  │  │  MCP Server  │  │  Optimization │  │  Real-time   │         │
│  │   (Typer)    │  │  (Stdio/HTTP)│  │    Agent     │  │   Updater    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                 │                   │
└─────────┼─────────────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │                 │
          └─────────────────┴────────┬────────┴─────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │      Application Layer (main.py) │
                    │  - App Context Management        │
                    │  - Session Logging               │
                    │  - Agent Initialization          │
                    │  - Tool Registration             │
                    └────────────────┬────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
┌─────────▼──────────┐    ┌──────────▼──────────┐    ┌─────────▼──────────┐
│  RAG Orchestrator  │    │  Cypher Generator   │    │   Graph Updater    │
│  (Pydantic AI)     │    │  (LLM Agent)        │    │   (GraphUpdater)   │
│  - Tool Coordination│   │  - NL to Cypher     │    │   - File Parsing   │
│  - Chat Loop       │    │  - Query Validation │    │   - AST Processing │
│  - Approval Flow   │    │  - Read-only Check  │    │   - Call Resolution│
└─────────┬──────────┘    └──────────┬──────────┘    └─────────┬──────────┘
          │                          │                          │
          │                          │                          │
┌─────────▼──────────────────────────▼──────────────────────────▼──────────┐
│                         Services Layer                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐              │
│  │ MemgraphIngestor│  │  QueryProtocol │  │ IngestorProtocol│             │
│  │ - Node/Rel Ops │  │ - Graph Queries│  │ - Data Ingestion│             │
│  │ - Batch Flush  │  │ - Export       │  │ - Constraints  │              │
│  │ - Transactions │  │ - Stats        │  │ - Indexes      │              │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘              │
└───────────┼───────────────────┼───────────────────┼──────────────────────┘
            │                   │                   │
            │                   │                   │
┌───────────▼───────┐  ┌────────▼───────┐  ┌───────▼──────────┐
│  Memgraph DB      │  │ Vector Store   │  │   Parser System  │
│  (Graph Database) │  │ (Embeddings)   │  │  (Tree-sitter)   │
│  - Nodes/Edges    │  │ - Memgraph     │  │  - Multi-language│
│  - Indexes        │  │ - Qdrant       │  │  - AST Cache     │
│  - Vector Search  │  │ - Similarity   │  │  - FQN Registry  │
└───────────────────┘  └────────────────┘  └──────────────────┘
```

## Detailed Component Architecture

### 1. Entry Points

```
┌────────────────────────────────────────────────────────────────┐
│                      Entry Points                               │
├────────────────────────────────────────────────────────────────┤
│  main.py                                                       │
│    │                                                           │
│    └─► codebase_rag.cli:app (Typer CLI)                       │
│         │                                                      │
│         ├─► start          (Interactive RAG + Graph Update)    │
│         ├─► index          (Protobuf Indexing)                 │
│         ├─► export         (Graph Export to JSON)              │
│         ├─► optimize       (AI Code Optimization)              │
│         ├─► mcp-server     (MCP Protocol Server)               │
│         ├─► graph-loader   (Load Exported Graph)               │
│         ├─► language       (Language Management)               │
│         ├─► doctor         (Health Check)                      │
│         └─► stats          (Graph Statistics)                  │
└────────────────────────────────────────────────────────────────┘
```

### 2. Core Processing Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                    Graph Updater Pipeline                           │
│                         (GraphUpdater)                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Pass 1: Structure Identification                                   │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ StructureProcessor                                    │          │
│  │ - Scan repository                                      │          │
│  │ - Identify packages, folders, files                   │          │
│  │ - Detect dependency files (pyproject.toml, etc.)      │          │
│  └──────────────────────────────────────────────────────┘          │
│                          │                                          │
│                          ▼                                          │
│  Pass 2: File Processing                                            │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ DefinitionProcessor                                   │          │
│  │ - Parse files with Tree-sitter                        │          │
│  │ - Extract functions, classes, methods                 │          │
│  │ - Build qualified names                               │          │
│  │ - Process imports/exports                             │          │
│  │ - Track inheritance relationships                     │          │
│  └──────────────────────────────────────────────────────┘          │
│                          │                                          │
│                          ▼                                          │
│  Pass 3: Call Resolution                                            │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ CallProcessor                                         │          │
│  │ - Resolve function calls                              │          │
│  │ - Type inference                                      │          │
│  │ - Build CALLS relationships                           │          │
│  │ - Handle method overrides                             │          │
│  └──────────────────────────────────────────────────────┘          │
│                          │                                          │
│                          ▼                                          │
│  Pass 4: Semantic Embeddings                                        │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Embedding Generation                                  │          │
│  │ - Extract source code                                 │          │
│  │ - Generate embeddings (768-dim)                       │          │
│  │ - Store in vector backend                             │          │
│  └──────────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────┘
```

### 3. Parser Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Parser System                                   │
│                    (codebase_rag/parsers/)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ProcessorFactory (Lazy Initialization)                             │
│    │                                                                 │
│    ├─► StructureProcessor                                           │
│    │   - identify_structure()                                        │
│    │   - process_generic_file()                                      │
│    │                                                                 │
│    ├─► DefinitionProcessor                                          │
│    │   - process_file()                                              │
│    │   - process_dependencies()                                      │
│    │   - process_all_method_overrides()                              │
│    │                                                                 │
│    ├─► ImportProcessor                                              │
│    │   - Resolve module imports                                      │
│    │   - Track dependencies                                          │
│    │                                                                 │
│    ├─► TypeInferenceEngine                                          │
│    │   - Infer types from calls                                      │
│    │   - Handle class inheritance                                    │
│    │                                                                 │
│    └─► CallProcessor                                                │
│        - process_calls_in_file()                                     │
│        - Resolve qualified names                                     │
│                                                                      │
│  Language Support (Tree-sitter Grammars):                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Python │ JavaScript │ TypeScript │ Rust │ Java │ C++ │ C │   │   │
│  │ Solidity │ Lua │ PHP │ Go │ Scala │ C# │ ...               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. RAG Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RAG Orchestrator Agent                          │
│                    (Pydantic AI Agent)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Agent Configuration:                                               │
│  - Model: Configurable (Google/OpenAI/Ollama)                       │
│  - System Prompt: RAG orchestration instructions                    │
│  - Tools: 10 specialized tools                                      │
│  - Output: str | DeferredToolRequests                               │
│                                                                      │
│  Available Tools:                                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 1. query_graph           - Natural language graph queries    │  │
│  │ 2. get_code_snippet      - Retrieve source by qualified name │  │
│  │ 3. read_file             - Read file contents                │  │
│  │ 4. create_file           - Create new files                  │  │
│  │ 5. replace_code          - Surgical code replacement         │  │
│  │ 6. list_directory        - List directory contents            │  │
│  │ 7. analyze_document      - Analyze PDFs/images               │  │
│  │ 8. execute_shell         - Run shell commands (sandboxed)    │  │
│  │ 9. semantic_search       - Vector similarity search          │  │
│  │ 10. get_function_source  - Get function by node ID           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Interactive Loop:                                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ User Question → Agent Thinking → Tool Calls → Approval?     │  │
│  │                              │                                │  │
│  │                      ┌───────┴───────┐                       │  │
│  │                      │               │                       │  │
│  │                   Approved        Denied                     │  │
│  │                      │               │                       │  │
│  │                      ▼               ▼                       │  │
│  │              Execute Tool     Request Feedback               │  │
│  │                      │               │                       │  │
│  │                      └───────┬───────┘                       │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │                    Generate Response                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5. Graph Database Schema

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Memgraph Knowledge Graph                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Node Types (from actual codebase):                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Label              │ Count  │ Key Properties                │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ Method             │ 3,637  │ qualified_name, name, path    │   │
│  │ Function           │ 1,855  │ qualified_name, name, path    │   │
│  │ Class              │   842  │ qualified_name, name, path    │   │
│  │ Module             │   529  │ qualified_name, name, path    │   │
│  │ File               │   508  │ path, name, extension         │   │
│  │ Folder             │    32  │ path, name, absolute_path     │   │
│  │ Package            │    19  │ qualified_name, name, path    │   │
│  │ Interface          │   ...  │ qualified_name, name          │   │
│  │ Enum               │   ...  │ qualified_name, name          │   │
│  │ Contract           │   ...  │ qualified_name, is_abstract   │   │
│  │ Library            │   ...  │ qualified_name, name          │   │
│  │ Event              │   ...  │ qualified_name, parameters    │   │
│  │ Modifier           │   ...  │ qualified_name, parameters    │   │
│  │ StateVariable      │   ...  │ qualified_name, type          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Relationships:                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Source          │ Relationship      │ Target                │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ Project/Package │ CONTAINS_*        │ Package/Folder/File   │   │
│  │ Module          │ DEFINES           │ Class/Function/etc.   │   │
│  │ Class           │ DEFINES_METHOD    │ Method                │   │
│  │ Module          │ IMPORTS/EXPORTS   │ Module/Class/Function │   │
│  │ Class           │ INHERITS          │ Class                 │   │
│  │ Class           │ IMPLEMENTS        │ Interface             │   │
│  │ Method          │ OVERRIDES         │ Method                │   │
│  │ Function/Method │ CALLS             │ Function/Method       │   │
│  │ Project         │ DEPENDS_ON_EXTERNAL│ ExternalPackage      │   │
│  │ Function/Method │ EMITS             │ Event                 │   │
│  │ Function/Method │ MODIFIED_BY       │ Modifier              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Indexes & Constraints:                                             │
│  - Unique constraints on qualified_name for most node types        │
│  - Indexes on frequently queried properties                        │
│  - Vector index for semantic search (768-dim embeddings)           │
└─────────────────────────────────────────────────────────────────────┘
```

### 6. Vector Store Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Vector Store Backend                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Backend Selection (configurable):                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ VECTOR_STORE_BACKEND = "memgraph" (default) or "qdrant"      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Option 1: Memgraph Native Vectors                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ - Vector index on Function/Method nodes                       │  │
│  │ - 768-dimensional embeddings                                  │  │
│  │ - Cosine similarity                                           │  │
│  │ - Integrated with graph queries                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Option 2: Qdrant Vector Database                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ - Separate Qdrant instance                                    │  │
│  │ - Collection: code_embeddings                                 │  │
│  │ - Payload: node_id, qualified_name                            │  │
│  │ - Cosine distance                                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Embedding Pipeline:                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 1. Extract source code from AST                               │  │
│  │ 2. Generate embedding (UnixCoder/other model)                 │  │
│  │ 3. Store with node_id reference                               │  │
│  │ 4. Cache embeddings locally                                   │  │
│  │ 5. Verify storage consistency                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 7. Data Flow Diagrams

#### Graph Update Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Repository │────►│ GraphUpdater │────►│ Memgraph     │
│  (File Sys) │     │              │     │ Database     │
└─────────────┘     └──────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Hash Cache   │
                    │ (.cgr-hash)  │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Vector Store │
                    │ (Embeddings) │
                    └──────────────┘
```

#### Query Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   User      │────►│ RAG Agent    │────►│ Cypher       │
│  Question   │     │ (Orchestrator)│    │ Generator    │
└─────────────┘     └──────────────┘     └──────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │  Memgraph    │
                                       │  Query       │
                                       └──────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │  Results +   │
                                       │  Code Snippets│
                                       └──────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │  AI Response │
                                       │  to User     │
                                       └──────────────┘
```

### 8. Configuration System

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Configuration Management                          │
│                      (codebase_rag/config.py)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Environment Variables (.env file):                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Database:                                                     │  │
│  │ - MEMGRAPH_HOST, MEMGRAPH_PORT, MEMGRAPH_USERNAME/PASSWORD   │  │
│  │ - MEMGRAPH_BATCH_SIZE                                        │  │
│  │                                                               │  │
│  │ Orchestrator Model:                                           │  │
│  │ - ORCHESTRATOR_PROVIDER (google/openai/ollama)               │  │
│  │ - ORCHESTRATOR_MODEL                                         │  │
│  │ - ORCHESTRATOR_API_KEY                                       │  │
│  │ - ORCHESTRATOR_ENDPOINT                                      │  │
│  │                                                               │  │
│  │ Cypher Model:                                                 │  │
│  │ - CYPHER_PROVIDER (google/openai/ollama)                     │  │
│  │ - CYPHER_MODEL                                               │  │
│  │ - CYPHER_API_KEY                                             │  │
│  │ - CYPHER_ENDPOINT                                            │  │
│  │                                                               │  │
│  │ Vector Store:                                                 │  │
│  │ - VECTOR_STORE_BACKEND (memgraph/qdrant)                     │  │
│  │ - QDRANT_URI, QDRANT_COLLECTION_NAME                         │  │
│  │ - MEMGRAPH_VECTOR_* settings                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Settings Class (AppConfig):                                        │
│  - Pydantic Settings with .env support                              │
│  - ModelConfig dataclass for LLM configuration                      │
│  - Validation for API keys                                          │
│  - Default values for all settings                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 9. Tool System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Tool System                                  │
│              (codebase_rag/tools/)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Tool Wrappers (Pydantic AI Tools):                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Tool Name              │ Handler Class      │ Function        │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ query_graph            │ -                  │ query_graph()   │  │
│  │ get_code_snippet       │ CodeRetriever      │ get_code()      │  │
│  │ read_file              │ FileReader         │ read_file()     │  │
│  │ create_file            │ FileWriter         │ create_file()   │  │
│  │ replace_code           │ FileEditor         │ replace_code()  │  │
│  │ list_directory         │ DirectoryLister    │ list_dir()      │  │
│  │ analyze_document       │ DocumentAnalyzer   │ analyze_doc()   │  │
│  │ execute_shell          │ ShellCommander     │ exec_shell()    │  │
│  │ semantic_search        │ -                  │ semantic_search()│ │
│  │ get_function_source    │ -                  │ get_function()  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Security Features:                                                 │
│  - Project root validation (prevent path traversal)                 │
│  - Shell command allowlist                                          │
│  - Edit confirmation workflow                                       │
│  - Read-only vs write operation distinction                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 10. MCP Server Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MCP Server Architecture                           │
│                  (codebase_rag/mcp/)                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Transport Modes:                                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ STDIO (default)          │ HTTP                               │  │
│  │ - Claude Code integration │ - REST API endpoint               │  │
│  │ - stdin/stdout communication │ - JSON-RPC over HTTP          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Available MCP Tools:                                               │
│  - list_projects: List indexed projects                             │
│  - delete_project: Remove project from graph                        │
│  - wipe_database: Clear entire database                             │
│  - index_repository: Parse and ingest repository                    │
│  - update_repository: Incremental update                            │
│  - query_code_graph: Natural language queries                       │
│  - get_code_snippet: Retrieve source code                           │
│  - surgical_replace_code: Code editing                              │
│  - read_file/write_file: File operations                            │
│  - list_directory: Directory listing                                │
│  - semantic_search: Vector search                                   │
│  - ask_agent: Full RAG pipeline                                     │
│                                                                      │
│  Integration:                                                       │
│  claude mcp add --transport stdio code-graph-rag \\                 │
│    --env TARGET_REPO_PATH=/path/to/repo \\                          │
│    --env CYPHER_PROVIDER=openai \\                                  │
│    --env CYPHER_MODEL=gpt-4 \\                                      │
│    -- uv run --directory /path/to/code-graph-rag \\                 │
│         code-graph-rag mcp-server                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Technology Stack                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Core Technologies:                                                 │
│  - Python 3.12+                                                     │
│  - Tree-sitter (AST parsing)                                        │
│  - Memgraph (Graph database with native vectors)                    │
│  - Pydantic AI (Agent framework)                                    │
│  - Typer (CLI framework)                                            │
│  - Rich (Terminal UI)                                               │
│                                                                      │
│  AI/ML:                                                             │
│  - Google Gemini (Cloud LLM)                                        │
│  - OpenAI GPT (Cloud LLM)                                           │
│  - Ollama (Local LLM)                                               │
│  - UnixCoder (Code embeddings)                                      │
│                                                                      │
│  Vector Storage:                                                    │
│  - Memgraph native vectors (default)                                │
│  - Qdrant (alternative)                                             │
│                                                                      │
│  Development Tools:                                                 │
│  - uv (Package manager)                                             │
│  - pytest (Testing)                                                 │
│  - ruff (Linting/formatting)                                        │
│  - Docker/Docker Compose                                            │
│                                                                      │
│  Integration:                                                       │
│  - MCP (Model Context Protocol)                                     │
│  - Claude Code                                                      │
│  - Git (Version control)                                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Patterns

1. **Lazy Initialization**: ProcessorFactory uses properties for lazy loading
2. **Protocol-based Design**: IngestorProtocol, QueryProtocol for abstraction
3. **Context Managers**: MemgraphIngestor uses __enter__/__exit__ for resource management
4. **Batch Processing**: Buffered node/relationship insertion with configurable batch sizes
5. **Incremental Updates**: Hash-based file change detection
6. **Trie-based Registry**: FunctionRegistryTrie for efficient qualified name lookups
7. **Bounded Cache**: LRU cache with memory limits for AST nodes
8. **Tool Approval Workflow**: Interactive confirmation for destructive operations
9. **Multi-backend Support**: Vector store abstraction for Memgraph/Qdrant
10. **Provider Pattern**: LLM provider abstraction for multiple AI backends

## Performance Optimizations

- Parallel batch flushing with ThreadPoolExecutor
- AST caching with memory-bounded LRU eviction
- Incremental graph updates via file hash comparison
- Batch vector embedding storage
- Connection pooling for Memgraph operations
- Lazy processor initialization
