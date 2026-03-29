from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

from rich.console import Console

from .constants import SupportedLanguage
from .types_defs import MCPHandlerType, MCPInputSchema, PropertyValue

if TYPE_CHECKING:
    from tree_sitter import Node


@dataclass
class SessionState:
    confirm_edits: bool = True
    log_file: Path | None = None
    cancelled: bool = False

    def reset_cancelled(self) -> None:
        self.cancelled = False


def _default_console() -> Console:
    return Console(width=None, force_terminal=True)


@dataclass
class AppContext:
    session: SessionState = field(default_factory=SessionState)
    console: Console = field(default_factory=_default_console)


@dataclass
class GraphNode:
    node_id: int
    labels: list[str]
    properties: dict[str, PropertyValue]


@dataclass
class GraphRelationship:
    from_id: int
    to_id: int
    type: str
    properties: dict[str, PropertyValue]


class FQNSpec(NamedTuple):
    scope_node_types: frozenset[str]
    function_node_types: frozenset[str]
    get_name: Callable[["Node"], str | None]
    file_to_module_parts: Callable[[Path, Path], list[str]]


@dataclass(frozen=True)
class LanguageSpec:
    language: SupportedLanguage | str
    file_extensions: tuple[str, ...]
    function_node_types: tuple[str, ...]
    class_node_types: tuple[str, ...]
    module_node_types: tuple[str, ...]
    call_node_types: tuple[str, ...] = ()
    import_node_types: tuple[str, ...] = ()
    import_from_node_types: tuple[str, ...] = ()
    name_field: str = "name"
    body_field: str = "body"
    package_indicators: tuple[str, ...] = ()
    function_query: str | None = None
    class_query: str | None = None
    call_query: str | None = None


@dataclass
class Dependency:
    name: str
    spec: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class MethodModifiersAndAnnotations:
    modifiers: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)


@dataclass
class ToolMetadata:
    name: str
    description: str
    input_schema: MCPInputSchema
    handler: MCPHandlerType
    returns_json: bool


@dataclass
class CodeChunk:
    """A semantic chunk of code split at natural boundaries.

    Attributes:
        content: The source code content of the chunk.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        chunk_type: Type of boundary where chunk was split.
        parent_scope: Optional scope name (e.g., class name) containing this chunk.
        parent_fqn: Fully qualified name of parent Function/Method.
        chunk_index: Position in chunk sequence (0-indexed).
        token_count: Approximate token count for this chunk.
        metadata: Additional metadata (e.g., original node type).
    """

    content: str
    start_line: int
    end_line: int
    chunk_type: Literal["class", "function", "method", "block", "statement", "truncated"]
    parent_scope: str | None = None
    parent_fqn: str | None = None
    chunk_index: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkedEmbeddingMetadata:
    """Metadata for chunked embeddings.

    Attributes:
        parent_fqn: Qualified name of parent Function/Method.
        chunk_index: Position in chunk sequence.
        total_chunks: Total number of chunks for this parent.
        start_line: Starting line number.
        end_line: Ending line number.
        chunk_type: Type of chunk boundary.
        is_summary: True if this is a summary chunk (hierarchical strategy).
    """

    parent_fqn: str
    chunk_index: int
    total_chunks: int
    start_line: int
    end_line: int
    chunk_type: str
    is_summary: bool = False


@dataclass
class EmbeddingResult:
    """Result of embedding generation.

    Attributes:
        embedding: The embedding vector.
        chunk: Optional CodeChunk if the code was split.
        metadata: Optional ChunkedEmbeddingMetadata for chunked embeddings.
        is_summary: True if this is a summary embedding (hierarchical strategy).
        child_embeddings: Optional list of child embeddings (hierarchical strategy).
    """

    embedding: list[float]
    chunk: CodeChunk | None = None
    metadata: ChunkedEmbeddingMetadata | None = None
    is_summary: bool = False
    child_embeddings: list[list[float]] | None = None


@dataclass
class TruncationResult:
    """Result of smart query result truncation.

    Attributes:
        results: The truncated list of result rows.
        tokens_used: Total tokens used in the truncated results.
        was_truncated: True if truncation occurred.
        dropped_count: Number of rows dropped.
        truncation_reason: Reason for truncation (token_limit, row_cap, none).
        dropped_rows: List of dropped rows for debugging.
    """

    results: list[dict]
    tokens_used: int
    was_truncated: bool
    dropped_count: int
    truncation_reason: Literal["token_limit", "row_cap", "none"]
    dropped_rows: list[dict] = field(default_factory=list)
