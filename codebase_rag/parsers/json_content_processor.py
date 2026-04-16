"""Processes arbitrary JSON configuration files into graph nodes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .. import constants as cs
from .. import logs as ls
from ..services import IngestorProtocol


class JsonContentProcessor:
    """Extracts JSON file content into graph nodes.

    Handles arbitrary JSON files (not canonical ingestion payloads) by creating
    JsonObject, JsonArray, JsonField, and JsonValue nodes with appropriate
    relationships.

    Qualified name convention:
        project_name.json.file_path_without_ext.field1.nested_field2

    This avoids collisions with Python modules (e.g., server.json vs server.py).
    """

    __slots__ = ("ingestor", "repo_path", "project_name")

    def __init__(
        self,
        ingestor: IngestorProtocol,
        repo_path: Path,
        project_name: str,
    ) -> None:
        self.ingestor = ingestor
        self.repo_path = repo_path
        self.project_name = project_name

    def process_json_file(self, filepath: Path) -> None:
        """Parse a JSON file and ingest its structure as graph nodes."""
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Skipping invalid JSON file {filepath}: {exc}")
            return

        relative_path = filepath.relative_to(self.repo_path).as_posix()
        file_stem = filepath.stem  # e.g., "server" from "server.json"

        # Build the root qualified name with json. namespace prefix
        root_qn = f"{self.project_name}.json.{file_stem}"

        # Create JsonObject node for root
        self.ingestor.ensure_node_batch(
            cs.NodeLabel.JSON_OBJECT,
            {
                cs.KEY_QUALIFIED_NAME: root_qn,
                cs.KEY_PATH: relative_path,
                cs.KEY_JSON_DEPTH: 0,
            },
        )

        # Create File -> JsonObject relationship
        self.ingestor.ensure_relationship_batch(
            (cs.NodeLabel.FILE, cs.KEY_PATH, relative_path),
            cs.RelationshipType.CONTAINS_JSON,
            (cs.NodeLabel.JSON_OBJECT, cs.KEY_QUALIFIED_NAME, root_qn),
        )

        # Recursively ingest JSON structure
        self._ingest_json_value(data, root_qn, relative_path, depth=0)

    def _ingest_json_value(
        self,
        value: Any,
        parent_qn: str,
        file_path: str,
        depth: int,
        key: str | None = None,
        index: int | None = None,
    ) -> None:
        """Recursively ingest a JSON value and its children."""
        if isinstance(value, dict):
            self._ingest_json_object(value, parent_qn, file_path, depth + 1, key, index)
        elif isinstance(value, list):
            self._ingest_json_array(value, parent_qn, file_path, depth + 1, key, index)
        else:
            self._ingest_json_scalar(value, parent_qn, file_path, depth + 1, key, index)

    def _ingest_json_object(
        self,
        obj: dict[str, Any],
        parent_qn: str,
        file_path: str,
        depth: int,
        key: str | None,
        index: int | None,
    ) -> None:
        """Ingest a JSON object's key-value pairs as JsonField nodes."""
        for k, v in obj.items():
            field_qn = f"{parent_qn}.{k}"

            # Classify the value for the field node
            value_str, value_type = self._classify_value(v)
            is_complex = isinstance(v, (dict, list))

            # Create JsonField node
            self.ingestor.ensure_node_batch(
                cs.NodeLabel.JSON_FIELD,
                {
                    cs.KEY_QUALIFIED_NAME: field_qn,
                    cs.KEY_PATH: file_path,
                    cs.KEY_JSON_KEY: k,
                    cs.KEY_JSON_VALUE: value_str if not is_complex else None,
                    cs.KEY_JSON_VALUE_TYPE: value_type if not is_complex else None,
                    cs.KEY_JSON_DEPTH: depth,
                },
            )

            # HAS_FIELD relationship from parent object to field
            self.ingestor.ensure_relationship_batch(
                (cs.NodeLabel.JSON_OBJECT, cs.KEY_QUALIFIED_NAME, parent_qn),
                cs.RelationshipType.HAS_FIELD,
                (cs.NodeLabel.JSON_FIELD, cs.KEY_QUALIFIED_NAME, field_qn),
            )

            # Recurse for nested values (dict or list)
            if is_complex:
                self._ingest_json_value(v, field_qn, file_path, depth, key=k)
            else:
                # Create JsonValue node and HAS_VALUE relationship for scalars
                self._create_scalar_value(value_str, value_type, field_qn, file_path, depth)

    def _ingest_json_array(
        self,
        arr: list[Any],
        parent_qn: str,
        file_path: str,
        depth: int,
        key: str | None,
        index: int | None,
    ) -> None:
        """Ingest a JSON array with its elements.

        FIX: Creates HAS_ELEMENT relationships for ALL array elements,
        including nested dicts/lists (the spec had a bug where it only
        created HAS_ELEMENT for scalar values).
        """
        array_qn = f"{parent_qn}._array"

        # Create JsonArray node
        self.ingestor.ensure_node_batch(
            cs.NodeLabel.JSON_ARRAY,
            {
                cs.KEY_QUALIFIED_NAME: array_qn,
                cs.KEY_PATH: file_path,
                cs.KEY_JSON_DEPTH: depth,
                cs.KEY_JSON_LENGTH: len(arr),
            },
        )

        # HAS_ELEMENT relationship from parent to array
        # Parent could be either JsonObject or JsonField
        parent_label = self._determine_parent_label(parent_qn)
        self.ingestor.ensure_relationship_batch(
            (parent_label, cs.KEY_QUALIFIED_NAME, parent_qn),
            cs.RelationshipType.HAS_ELEMENT,
            (cs.NodeLabel.JSON_ARRAY, cs.KEY_QUALIFIED_NAME, array_qn),
        )

        # Process each element using dot-notation indices (avoid [] special chars)
        for i, item in enumerate(arr):
            element_qn = f"{array_qn}.{i}"
            if isinstance(item, dict):
                # Nested object: create JsonField wrapper, then recurse
                field_qn = element_qn
                self.ingestor.ensure_node_batch(
                    cs.NodeLabel.JSON_FIELD,
                    {
                        cs.KEY_QUALIFIED_NAME: field_qn,
                        cs.KEY_PATH: file_path,
                        cs.KEY_JSON_KEY: str(i),
                        cs.KEY_JSON_VALUE: None,
                        cs.KEY_JSON_VALUE_TYPE: None,
                        cs.KEY_JSON_DEPTH: depth,
                    },
                )
                self.ingestor.ensure_relationship_batch(
                    (cs.NodeLabel.JSON_ARRAY, cs.KEY_QUALIFIED_NAME, array_qn),
                    cs.RelationshipType.HAS_ELEMENT,
                    (cs.NodeLabel.JSON_FIELD, cs.KEY_QUALIFIED_NAME, field_qn),
                )
                self._ingest_json_value(item, field_qn, file_path, depth + 1, index=i)

            elif isinstance(item, list):
                # Nested array: recurse (HAS_ELEMENT created in recursive call)
                self._ingest_json_array(item, array_qn, file_path, depth + 1, index=i)

            else:
                # Scalar value: create JsonValue and HAS_ELEMENT
                value_str, value_type = self._classify_value(item)
                self.ingestor.ensure_node_batch(
                    cs.NodeLabel.JSON_VALUE,
                    {
                        cs.KEY_QUALIFIED_NAME: element_qn,
                        cs.KEY_PATH: file_path,
                        cs.KEY_JSON_VALUE: value_str,
                        cs.KEY_JSON_VALUE_TYPE: value_type,
                        cs.KEY_JSON_DEPTH: depth + 1,
                    },
                )
                self.ingestor.ensure_relationship_batch(
                    (cs.NodeLabel.JSON_ARRAY, cs.KEY_QUALIFIED_NAME, array_qn),
                    cs.RelationshipType.HAS_ELEMENT,
                    (cs.NodeLabel.JSON_VALUE, cs.KEY_QUALIFIED_NAME, element_qn),
                )

    def _ingest_json_scalar(
        self,
        value: Any,
        parent_qn: str,
        file_path: str,
        depth: int,
        key: str | None,
        index: int | None,
    ) -> None:
        """Ingest a scalar JSON value (string, number, boolean, null)."""
        value_str, value_type = self._classify_value(value)
        # Use the parent_qn directly as the field QN for module-level scalars
        self._create_scalar_value(value_str, value_type, parent_qn, file_path, depth)

    def _create_scalar_value(
        self,
        value_str: str,
        value_type: str,
        parent_qn: str,
        file_path: str,
        depth: int,
    ) -> None:
        """Create a JsonValue node with a distinct QN to avoid collision with JsonField."""
        # FIX: Use ._value suffix to avoid QN collision with the parent JsonField
        value_qn = f"{parent_qn}._value"

        self.ingestor.ensure_node_batch(
            cs.NodeLabel.JSON_VALUE,
            {
                cs.KEY_QUALIFIED_NAME: value_qn,
                cs.KEY_PATH: file_path,
                cs.KEY_JSON_VALUE: value_str,
                cs.KEY_JSON_VALUE_TYPE: value_type,
                cs.KEY_JSON_DEPTH: depth,
            },
        )

        # HAS_VALUE relationship from JsonField to JsonValue
        self.ingestor.ensure_relationship_batch(
            (cs.NodeLabel.JSON_FIELD, cs.KEY_QUALIFIED_NAME, parent_qn),
            cs.RelationshipType.HAS_VALUE,
            (cs.NodeLabel.JSON_VALUE, cs.KEY_QUALIFIED_NAME, value_qn),
        )

    @staticmethod
    def _determine_parent_label(parent_qn: str) -> cs.NodeLabel:
        """Determine the node label based on the QN pattern.

        JsonArray QNs end with ._array, so if the parent_qn ends with ._array,
        the parent is also a JsonArray. Otherwise it's a JsonObject or JsonField.
        """
        if parent_qn.endswith("._array"):
            return cs.NodeLabel.JSON_ARRAY
        # Check if it's a field (contains a key segment) vs object
        # For simplicity, default to JSON_OBJECT for object parents
        return cs.NodeLabel.JSON_OBJECT

    @staticmethod
    def _classify_value(value: Any) -> tuple[str, str]:
        """Classify a JSON scalar value into (stringified_value, value_type)."""
        if value is None:
            return "null", "null"
        if isinstance(value, bool):
            return str(value).lower(), "boolean"
        if isinstance(value, (int, float)):
            return str(value), "number"
        return str(value), "string"
