from enum import StrEnum


class CLICommandName(StrEnum):
    START = "start"
    INDEX = "index"
    EXPORT = "export"
    OPTIMIZE = "optimize"
    MCP_SERVER = "mcp-server"
    GRAPH_LOADER = "graph-loader"
    LANGUAGE = "language"
    DOCTOR = "doctor"
    STATS = "stats"
    # Document GraphRAG commands
    QUERY_DOCS = "query-docs"
    QUERY_ALL = "query-all"
    VALIDATE_SPEC = "validate-spec"
    VALIDATE_DOC = "validate-doc"
    INDEX_DOCS = "index-docs"


APP_DESCRIPTION = (
    "An accurate Retrieval-Augmented Generation (RAG) system that analyzes "
    "multi-language codebases using Tree-sitter, builds comprehensive knowledge "
    "graphs, and enables natural language querying of codebase structure and relationships."
)

CMD_START = (
    "Start interactive chat session with your codebase. "
    "Supports both code and document GraphRAG. "
    "Use --with-docs to enable document queries. "
    "Use --index-docs or --index-all to index before chatting. "
    "Use --mode to specify query mode (default: code_only)."
)
CMD_INDEX = "Index codebase to protobuf files for offline use"
CMD_EXPORT = "Export knowledge graph from Memgraph to JSON file"
CMD_OPTIMIZE = "AI-guided codebase optimization session"
CMD_MCP_SERVER = "Start the MCP server for Claude Code integration"
CMD_GRAPH_LOADER = "Load and display summary of exported graph JSON"
CMD_LANGUAGE = "Manage language grammars (add, remove, list)"
CMD_DOCTOR = "Verify that all dependencies and configurations are properly set up"
CMD_STATS = "Display node and relationship statistics for the indexed graph"
# Document GraphRAG commands
CMD_QUERY_DOCS = "Query the document graph using natural language"
CMD_QUERY_ALL = "Query both code and document graphs, merge results"
CMD_VALIDATE_SPEC = "Validate code against a specification document"
CMD_VALIDATE_DOC = "Validate documentation against actual code"
CMD_INDEX_DOCS = "Index documents into the document graph"

CMD_LANGUAGE_GROUP = "CLI for managing language grammars"
CMD_LANGUAGE_ADD = "Add a new language grammar to the project."
CMD_LANGUAGE_LIST = "List all currently configured languages."
CMD_LANGUAGE_REMOVE = "Remove a language from the project."
CMD_LANGUAGE_CLEANUP = "Clean up orphaned git modules that weren't properly removed."

HELP_BATCH_SIZE = "Number of buffered nodes/relationships before flushing to Memgraph"
HELP_MEMGRAPH_HOST = "Memgraph host"
HELP_MEMGRAPH_PORT = "Memgraph port"
HELP_ORCHESTRATOR = (
    "Specify orchestrator as provider:model "
    "(e.g., ollama:llama3.2, openai:gpt-4, google:gemini-3.1-pro-preview)"
)
HELP_CYPHER_MODEL = (
    "Specify cypher model as provider:model "
    "(e.g., ollama:codellama, google:gemini-3-flash-preview)"
)
HELP_NO_CONFIRM = "Disable confirmation prompts for edit operations"
HELP_YOLO = "Disable all interactive confirmations (auto-approve all tool calls)"

HELP_REPO_PATH_RETRIEVAL = "Path to the target repository for code retrieval"
HELP_REPO_PATH_INDEX = "Path to the target repository to index."
HELP_REPO_PATH_OPTIMIZE = "Path to the repository to optimize"
HELP_REPO_PATH_WATCH = "Path to the repository to watch."
HELP_VERSION = "Show the version and exit."

HELP_DEBOUNCE = "Debounce delay in seconds. Set to 0 to disable debouncing."
HELP_MAX_WAIT = (
    "Maximum wait time in seconds before forcing an update during continuous edits."
)

HELP_UPDATE_GRAPH = "Update the knowledge graph by parsing the repository"
HELP_CLEAN_DB = "Clean the database before updating (use when adding first repo)"
HELP_OUTPUT_GRAPH = "Export graph to JSON file after updating (requires --update-graph)"
HELP_OUTPUT_PATH = "Output file path for the exported graph"
HELP_OUTPUT_PROTO_DIR = (
    "Required. Path to the output directory for the protobuf index file(s)."
)
HELP_SPLIT_INDEX = "Write index to separate nodes.bin and relationships.bin files."
HELP_FORMAT_JSON = "Export in JSON format"
HELP_LANGUAGE_ARG = (
    "Programming language to optimize for (e.g., python, java, javascript, cpp)"
)
HELP_REFERENCE_DOC = "Path to reference document/book for optimization guidance"
HELP_GRAPH_FILE = "Path to the exported graph JSON file"
HELP_EXPORTED_GRAPH_FILE = "Path to the exported_graph.json file."

HELP_GRAMMAR_URL = (
    "URL to the tree-sitter grammar repository. If not provided, "
    "will use https://github.com/tree-sitter/tree-sitter-<language_name>"
)
HELP_KEEP_SUBMODULE = "Keep the git submodule (default: remove it)"

HELP_PROJECT_NAME = (
    "Override the project name used as qualified-name prefix for all nodes. "
    "Defaults to the repo directory name."
)
HELP_EXCLUDE_PATTERNS = (
    "Additional directories to exclude from indexing. Can be specified multiple times."
)
HELP_INTERACTIVE_SETUP = (
    "Show interactive prompt to select which detected directories to keep. "
    "Without this flag, all directories matching ignore patterns are automatically excluded."
)

HELP_ASK_AGENT = (
    "Run a single query in non-interactive mode and exit. "
    "Output is sent to stdout, useful for scripting."
)

HELP_MCP_TRANSPORT = "Transport mode: 'stdio' (default) or 'http'"
HELP_MCP_HTTP_HOST = (
    "Host to bind the HTTP server — only used when --transport http (default: 0.0.0.0)"
)
HELP_MCP_HTTP_PORT = (
    "Port to bind the HTTP server — only used when --transport http (default: 8080)"
)

# Document GraphRAG help text
HELP_QUERY = "Natural language query"
HELP_TOP_K = "Maximum number of results to return"
HELP_SPEC_PATH = "Path to the specification document"
HELP_DOC_PATH = "Path to the document to validate"
HELP_SCOPE = "Scope of validation: 'all', 'sections', or 'claims'"
HELP_MAX_COST = "Maximum cost budget for validation in USD"
HELP_DRY_RUN = "Estimate cost without running validation"
HELP_CLEAN_DOC_DB = "Clean the document database before indexing (use when adding first documents)"

# Unified start command help text
HELP_WITH_DOCS = (
    "Connect to document graph (DOC_MEMGRAPH_HOST:DOC_MEMGRAPH_PORT) "
    "in addition to code graph. Required for document queries."
)
HELP_INDEX_DOCS = (
    "Index documents from --repo-path into document graph before starting chat. "
    "Implies --with-docs. Uses DocumentGraphUpdater with version caching."
)
HELP_INDEX_ALL = (
    "Index both code (--update-graph) and documents (--index-docs) before starting. "
    "Convenience flag for first-time setup or major updates."
)
HELP_DOC_WORKSPACE = (
    "Workspace identifier for multi-tenant document graphs. "
    "Isolates document data between projects. Must match workspace used during indexing."
)
HELP_MODE = (
    "Query routing mode. Options:\n"
    "  - code_only: Query code graph only (default)\n"
    "  - document_only: Query document graph only\n"
    "  - both_merged: Query both graphs, merge results with attribution\n"
    "  - code_vs_doc: Validate code against documentation (doc is truth)\n"
    "  - doc_vs_code: Validate documentation against code (code is truth)\n"
    "\nNote: Document graph requires --with-docs flag."
)
HELP_CHECK_FRESHNESS = (
    "Check if indexed graphs are up-to-date with repository. "
    "If stale, prompts to re-index. Disable with --no-check-freshness for faster startup."
)
HELP_INDEX_TIMEOUT = "Maximum seconds for indexing operations (default: 300s)"

CLI_COMMANDS: dict[CLICommandName, str] = {
    CLICommandName.START: CMD_START,
    CLICommandName.INDEX: CMD_INDEX,
    CLICommandName.EXPORT: CMD_EXPORT,
    CLICommandName.OPTIMIZE: CMD_OPTIMIZE,
    CLICommandName.MCP_SERVER: CMD_MCP_SERVER,
    CLICommandName.GRAPH_LOADER: CMD_GRAPH_LOADER,
    CLICommandName.LANGUAGE: CMD_LANGUAGE,
    CLICommandName.DOCTOR: CMD_DOCTOR,
    CLICommandName.STATS: CMD_STATS,
    # Document GraphRAG commands
    CLICommandName.QUERY_DOCS: CMD_QUERY_DOCS,
    CLICommandName.QUERY_ALL: CMD_QUERY_ALL,
    CLICommandName.VALIDATE_SPEC: CMD_VALIDATE_SPEC,
    CLICommandName.VALIDATE_DOC: CMD_VALIDATE_DOC,
    CLICommandName.INDEX_DOCS: CMD_INDEX_DOCS,
}
