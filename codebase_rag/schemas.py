from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .types_defs import ResultRow


def _normalize_value(val: Any) -> Any:
    """Recursively normalize a value to ensure Pydantic compatibility.

    Converts any non-standard types to strings while preserving the
    structure of nested lists and dicts.
    """
    if val is None:
        return None
    if isinstance(val, str | int | float | bool):
        return val
    if isinstance(val, list):
        return [_normalize_value(item) for item in val]
    if isinstance(val, dict):
        return {k: _normalize_value(v) for k, v in val.items()}
    # Convert any other type to string
    return str(val)


class QueryGraphData(BaseModel):
    query_used: str
    results: list[ResultRow]
    summary: str

    @field_validator("results", mode="before")
    @classmethod
    def _format_results(cls, v: list[ResultRow] | None) -> list[ResultRow]:
        if not isinstance(v, list):
            return []

        clean_results: list[ResultRow] = []
        for row in v:
            if not isinstance(row, dict):
                continue
            clean_row: ResultRow = {k: _normalize_value(val) for k, val in row.items()}
            clean_results.append(clean_row)
        return clean_results

    model_config = ConfigDict(extra="forbid")


class CodeSnippet(BaseModel):
    qualified_name: str
    source_code: str
    file_path: str
    line_start: int
    line_end: int
    docstring: str | None = None
    found: bool = True
    error_message: str | None = None


class ShellCommandResult(BaseModel):
    return_code: int
    stdout: str
    stderr: str


class EditResult(BaseModel):
    file_path: str
    success: bool = True
    error_message: str | None = None

    @model_validator(mode="after")
    def _set_success_on_error(self) -> EditResult:
        if self.error_message is not None:
            self.success = False
        return self


class FileReadResult(BaseModel):
    file_path: str
    content: str | None = None
    error_message: str | None = None


class FileCreationResult(BaseModel):
    file_path: str
    success: bool = True
    error_message: str | None = None

    @model_validator(mode="after")
    def _set_success_on_error(self) -> FileCreationResult:
        if self.error_message is not None:
            self.success = False
        return self


class HealthCheckResult(BaseModel):
    name: str
    passed: bool
    message: str
    error: str | None = None


class JSONEntity(BaseModel):
    id: str | None = None
    name: str
    type: str | None = None
    labels: list[str] | None = None
    operation: str | None = None
    last_updated: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class JSONRelationship(BaseModel):
    id: str | None = None
    source: str
    target: str
    relationship: str
    operation: str | None = None
    last_updated: str | None = None
    confidence: float | None = None
    explanation: str | None = None
    isInferred: bool | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class JSONMetadata(BaseModel):
    dataset_id: str
    source: str | None = None
    created_at: str | None = None
    default_entity_labels: list[str] | None = None
    operation: str | None = None
    last_updated: str | None = None


# JSONInputSchema REMOVED:
# Deprecated in favor of direct validation against ingestion_schema.json
# as the single source of truth for input validation


class IngestionResult(BaseModel):
    dataset_id: str
    # New fields for file tracking
    files_processed: int = 0
    files_skipped: int = 0
    # Existing fields unchanged
    entities_processed: int = 0
    entities_ingested: int = 0
    entities_updated: int = 0
    entities_deleted: int = 0
    entities_skipped: int = 0
    entities_failed: int = 0
    relationships_processed: int = 0
    relationships_ingested: int = 0
    relationships_updated: int = 0
    relationships_deleted: int = 0
    relationships_skipped: int = 0
    relationships_failed: int = 0
    errors: list[str] = Field(default_factory=list)
    dry_run: bool = False


class UpdateResult(IngestionResult):
    """Result of an incremental update event from streaming source."""

    operation: str = "add"
    event_id: str | None = None
    processed_at: str | None = None
