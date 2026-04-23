from typing import Protocol, runtime_checkable

from ..types_defs import PropertyDict, PropertyValue, ResultRow
from .error_guidance import (
    CachedGuidanceGenerator,
    ErrorContext,
    ErrorGuidance,
    LLMErrorGuidance,
    UserExpertiseLevel,
    format_user_error,
)
from .failure_classifier import (
    FailureClassification,
    FailureType,
    classify_memgraph_failure,
)


@runtime_checkable
class IngestorProtocol(Protocol):
    def ensure_node(self, label: str, properties: PropertyDict) -> None: ...

    def ensure_node_batch(self, label: str, properties: PropertyDict) -> None: ...

    def ensure_relationship_batch(
        self,
        from_spec: tuple[str, str, PropertyValue],
        rel_type: str,
        to_spec: tuple[str, str, PropertyValue],
        properties: PropertyDict | None = None,
    ) -> None: ...

    def ensure_edge(
        self,
        rel_type: str,
        from_identifier: str,
        to_identifier: str,
        properties: PropertyDict | None = None,
    ) -> None: ...

    def flush_all(self) -> None: ...


@runtime_checkable
class QueryProtocol(Protocol):
    def fetch_all(
        self, query: str, params: PropertyDict | None = None
    ) -> list[ResultRow]: ...

    async def fetch_all_async(
        self, query: str, params: PropertyDict | None = None
    ) -> list[ResultRow]: ...

    def execute_write(self, query: str, params: PropertyDict | None = None) -> None: ...


__all__ = [
    "IngestorProtocol",
    "QueryProtocol",
    "ErrorContext",
    "ErrorGuidance",
    "LLMErrorGuidance",
    "CachedGuidanceGenerator",
    "UserExpertiseLevel",
    "format_user_error",
    "FailureClassification",
    "FailureType",
    "classify_memgraph_failure",
]
