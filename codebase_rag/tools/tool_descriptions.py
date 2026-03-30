from __future__ import annotations

from enum import StrEnum

from codebase_rag.constants import MCPToolName


class AgenticToolName(StrEnum):
    QUERY_GRAPH = "query_graph"
    READ_FILE = "read_file"
    CREATE_FILE = "create_file"
    REPLACE_CODE = "replace_code"
    LIST_DIRECTORY = "list_directory"
    ANALYZE_DOCUMENT = "analyze_document"
    EXECUTE_SHELL = "execute_shell"
    SEMANTIC_SEARCH = "semantic_search"
    GET_FUNCTION_SOURCE = "get_function_source"
    GET_CODE_SNIPPET = "get_code_snippet"


ANALYZE_DOCUMENT = (
    "Analyzes documents (PDFs, images) to answer questions about their content."
)

CODEBASE_QUERY = (
    "Query the codebase knowledge graph using natural language questions. "
    "Ask in plain English about classes, functions, methods, dependencies, or code structure. "
    "Examples: 'Find all functions that call each other', "
    "'What classes are in the user module', "
    "'Show me functions with the longest call chains'."
)

DIRECTORY_LISTER = "Lists the contents of a directory to explore the codebase."

FILE_WRITER = (
    "Creates a new file with content. IMPORTANT: Check file existence first! "
    "Overwrites completely WITHOUT showing diff. "
    "Use only for new files, not existing file modifications."
)

SHELL_COMMAND = (
    "Executes shell commands from allowlist. "
    "Read-only commands run without approval; write operations require user confirmation. "
    "IMPORTANT: Shell redirect operators (> >> < << 2>/dev/null 2>&1 etc.) are NOT supported "
    "because direct process execution cannot interpret shell syntax."
)

CODE_RETRIEVAL = (
    "Retrieves the source code for a specific function, class, or method "
    "using its full qualified name."
)

SEMANTIC_SEARCH = (
    "Performs a semantic search for functions based on a natural language query "
    "describing their purpose, returning a list of potential matches with similarity scores."
)

GET_FUNCTION_SOURCE = (
    "Retrieves the source code for a specific function or method using its internal node ID, "
    "typically obtained from a semantic search result."
)

FILE_READER = (
    "Reads the content of text-based files. "
    "For documents like PDFs or images, use the 'analyze_document' tool instead."
)

FILE_EDITOR = (
    "Surgically replaces specific code blocks in files. "
    "Requires exact target code and replacement. "
    "Only modifies the specified block, leaving rest of file unchanged. "
    "True surgical patching."
)

# (H) MCP tool descriptions
MCP_LIST_PROJECTS = (
    "List all indexed projects in the knowledge graph database. "
    "Returns a list of project names that have been indexed."
)

MCP_DELETE_PROJECT = (
    "Delete a specific project from the knowledge graph database. "
    "This removes all nodes associated with the project while preserving other projects. "
    "Use list_projects first to see available projects."
)

MCP_WIPE_DATABASE = (
    "WARNING: Completely wipe the entire database, removing ALL indexed projects. "
    "This cannot be undone. Use delete_project for removing individual projects."
)

MCP_INDEX_REPOSITORY = (
    "WARNING: Clears all data for the current project including its embeddings. "
    "Parse and ingest the repository into the Memgraph knowledge graph. "
    "Use update_repository for incremental updates. Only use when explicitly requested."
)

MCP_UPDATE_REPOSITORY = (
    "Update the repository in the Memgraph knowledge graph without clearing existing data. "
    "Use this for incremental updates."
)

MCP_QUERY_CODE_GRAPH = (
    "Query the codebase knowledge graph using natural language. "
    "Use semantic_search unless you know the exact names of classes/functions you are searching for. "
    "Ask questions like 'What functions call UserService.create_user?' or "
    "'Show me all classes that implement the Repository interface'."
)

MCP_GET_CODE_SNIPPET = (
    "Retrieve source code for a function, class, or method by its qualified name. "
    "Returns the source code, file path, line numbers, and docstring."
)

MCP_SURGICAL_REPLACE_CODE = (
    "Surgically replace an exact code block in a file using diff-match-patch. "
    "Only modifies the exact target block, leaving the rest unchanged."
)

MCP_READ_FILE = (
    "Read the contents of a file from the project. Supports pagination for large files."
)

MCP_WRITE_FILE = "Write content to a file, creating it if it doesn't exist."

MCP_LIST_DIRECTORY = "List contents of a directory in the project."

MCP_SEMANTIC_SEARCH = (
    "Performs a semantic search for functions based on a natural language query "
    "describing their purpose, returning a list of potential matches with similarity scores. "
    "Requires the 'semantic' extra to be installed."
)

MCP_PARAM_PROJECT_NAME = "Name of the project to delete (e.g., 'my-project')"
MCP_PARAM_CONFIRM = "Must be true to confirm the wipe operation"
MCP_PARAM_NATURAL_LANGUAGE_QUERY = "Your question in plain English about the codebase"
MCP_PARAM_QUALIFIED_NAME = (
    "Fully qualified name (e.g., 'app.services.UserService.create_user')"
)
MCP_PARAM_FILE_PATH = "Relative path to the file from project root"
MCP_PARAM_TARGET_CODE = "Exact code block to replace"
MCP_PARAM_REPLACEMENT_CODE = "New code to insert"
MCP_PARAM_OFFSET = "Line number to start reading from (0-based, optional)"
MCP_PARAM_LIMIT = "Maximum number of lines to read (optional)"
MCP_PARAM_CONTENT = "Content to write to the file"
MCP_PARAM_DIRECTORY_PATH = "Relative path to directory from project root (default: '.')"
MCP_PARAM_TOP_K = "Max number of results to return (optional, default: 5)"
MCP_PARAM_QUESTION = (
    "A question about the codebase, architecture, functionality, or code relationships"
)

MCP_ASK_AGENT = (
    "Ask the Code Graph RAG agent a question about the codebase. "
    "Uses the full RAG pipeline to analyze the code graph and provide a detailed answer. "
    "Use this for general questions about architecture, functionality, and code relationships."
)

MCP_GET_EMBEDDING_STATUS = (
    "Get the current embedding provider configuration and status. "
    "Returns the current provider, model, dimension, and available providers."
)

MCP_SET_EMBEDDING_PROVIDER = (
    "Switch to a different embedding provider. "
    "Supported providers: local, openai, google, ollama. "
    "Optionally re-embed all vectors after switching. "
    "WARNING: Changing providers may require re-embedding for consistent results."
)

MCP_PARAM_EMBEDDING_PROVIDER = "Embedding provider name (local, openai, google, ollama)"
MCP_PARAM_EMBEDDING_MODEL = "Model identifier (e.g., text-embedding-3-small, nomic-embed-text)"
MCP_PARAM_EMBEDDING_DIMENSION = "Optional dimension override (auto-detected from model by default)"
MCP_PARAM_EMBEDDING_API_KEY = "API key for external providers (optional if set in environment)"
MCP_PARAM_EMBEDDING_ENDPOINT = "Custom endpoint URL (optional)"
MCP_PARAM_REEMBED = "If true, re-embed all vectors after switching providers (default: false)"

# Document GraphRAG tool descriptions
MCP_QUERY_DOCUMENT_GRAPH = (
    "Query the DOCUMENT graph/vector ONLY. "
    "Use for questions about documentation, specifications, and textual content. "
    "Returns relevant document sections and chunks."
)

MCP_QUERY_BOTH_GRAPHS = (
    "Query BOTH code and document graphs, merge results. "
    "Use for comprehensive searches spanning code and documentation. "
    "Results are labeled with their source (code_graph or document_graph)."
)

MCP_VALIDATE_CODE_AGAINST_SPEC = (
    "Validate CODE against DOCUMENT specifications. "
    "Checks if the implementation matches the specification document. "
    "Returns validation report with discrepancies."
)

MCP_VALIDATE_DOC_AGAINST_CODE = (
    "Validate DOCUMENT against actual CODE. "
    "Checks if documentation accurately reflects the current code. "
    "Identifies outdated or incorrect documentation."
)

MCP_INDEX_DOCUMENTS = (
    "Index documents into the document graph. "
    "Parses and ingests markdown, PDF, DOCX, and text files. "
    "Creates Document, Section, and Chunk nodes with embeddings."
)

MCP_PARAM_SPEC_DOCUMENT_PATH = "Path to the specification document to validate against"
MCP_PARAM_DOCUMENT_PATH = "Path to the document to validate"
MCP_PARAM_SCOPE = "Scope of validation: 'all', 'sections', or 'claims'"
MCP_PARAM_MAX_COST_USD = "Maximum cost budget for validation in USD (default: 0.50)"
MCP_PARAM_DRY_RUN = "If true, only estimate cost without running validation"


MCP_TOOLS: dict[MCPToolName, str] = {
    MCPToolName.LIST_PROJECTS: MCP_LIST_PROJECTS,
    MCPToolName.DELETE_PROJECT: MCP_DELETE_PROJECT,
    MCPToolName.WIPE_DATABASE: MCP_WIPE_DATABASE,
    MCPToolName.INDEX_REPOSITORY: MCP_INDEX_REPOSITORY,
    MCPToolName.UPDATE_REPOSITORY: MCP_UPDATE_REPOSITORY,
    MCPToolName.QUERY_CODE_GRAPH: MCP_QUERY_CODE_GRAPH,
    MCPToolName.GET_CODE_SNIPPET: MCP_GET_CODE_SNIPPET,
    MCPToolName.SURGICAL_REPLACE_CODE: MCP_SURGICAL_REPLACE_CODE,
    MCPToolName.READ_FILE: MCP_READ_FILE,
    MCPToolName.WRITE_FILE: MCP_WRITE_FILE,
    MCPToolName.LIST_DIRECTORY: MCP_LIST_DIRECTORY,
    MCPToolName.SEMANTIC_SEARCH: MCP_SEMANTIC_SEARCH,
    MCPToolName.ASK_AGENT: MCP_ASK_AGENT,
    MCPToolName.GET_EMBEDDING_STATUS: MCP_GET_EMBEDDING_STATUS,
    MCPToolName.SET_EMBEDDING_PROVIDER: MCP_SET_EMBEDDING_PROVIDER,
    # Document GraphRAG tools
    MCPToolName.QUERY_DOCUMENT_GRAPH: MCP_QUERY_DOCUMENT_GRAPH,
    MCPToolName.QUERY_BOTH_GRAPHS: MCP_QUERY_BOTH_GRAPHS,
    MCPToolName.VALIDATE_CODE_AGAINST_SPEC: MCP_VALIDATE_CODE_AGAINST_SPEC,
    MCPToolName.VALIDATE_DOC_AGAINST_CODE: MCP_VALIDATE_DOC_AGAINST_CODE,
    MCPToolName.INDEX_DOCUMENTS: MCP_INDEX_DOCUMENTS,
}

AGENTIC_TOOLS: dict[AgenticToolName, str] = {
    AgenticToolName.QUERY_GRAPH: CODEBASE_QUERY,
    AgenticToolName.READ_FILE: FILE_READER,
    AgenticToolName.CREATE_FILE: FILE_WRITER,
    AgenticToolName.REPLACE_CODE: FILE_EDITOR,
    AgenticToolName.LIST_DIRECTORY: DIRECTORY_LISTER,
    AgenticToolName.ANALYZE_DOCUMENT: ANALYZE_DOCUMENT,
    AgenticToolName.EXECUTE_SHELL: SHELL_COMMAND,
    AgenticToolName.SEMANTIC_SEARCH: SEMANTIC_SEARCH,
    AgenticToolName.GET_FUNCTION_SOURCE: GET_FUNCTION_SOURCE,
    AgenticToolName.GET_CODE_SNIPPET: CODE_RETRIEVAL,
}
