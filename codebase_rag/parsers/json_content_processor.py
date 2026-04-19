"""Processes arbitrary JSON configuration files into graph nodes."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from loguru import logger

from .. import constants as cs
from ..services import IngestorProtocol


class JsonContentProcessor:
    """Extracts arbitrary JSON file content into code-graph nodes."""

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
        try:
            with open(filepath, encoding=cs.ENCODING_UTF8) as file_handle:
                data = json.load(file_handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Skipping invalid JSON file {filepath}: {exc}")
            return

        relative_path = filepath.relative_to(self.repo_path)
        relative_path_str = relative_path.as_posix()
        base_qn = self._build_base_qualified_name(relative_path)
        root_label, root_qn = self._ingest_root_value(
            data,
            base_qn,
            relative_path_str,
        )
        self.ingestor.ensure_relationship_batch(
            (cs.NodeLabel.FILE, cs.KEY_PATH, relative_path_str),
            cs.RelationshipType.CONTAINS_JSON,
            (root_label, cs.KEY_QUALIFIED_NAME, root_qn),
        )

    def _ingest_root_value(
        self,
        value: Any,
        base_qn: str,
        file_path: str,
    ) -> tuple[cs.NodeLabel, str]:
        if isinstance(value, dict):
            return self._ingest_object_node(f"{base_qn}._object", value, file_path, 0)
        if isinstance(value, list):
            return self._ingest_array_node(f"{base_qn}._array", value, file_path, 0)
        return self._ingest_scalar_node(f"{base_qn}._value", value, file_path, 0)

    def _ingest_object_node(
        self,
        object_qn: str,
        obj: dict[str, Any],
        file_path: str,
        depth: int,
    ) -> tuple[cs.NodeLabel, str]:
        self.ingestor.ensure_node_batch(
            cs.NodeLabel.JSON_OBJECT,
            {
                cs.KEY_QUALIFIED_NAME: object_qn,
                cs.KEY_NAME: object_qn.split(cs.SEPARATOR_DOT)[-1],
                cs.KEY_PATH: file_path,
                cs.KEY_JSON_DEPTH: depth,
            },
        )

        for raw_key, raw_value in obj.items():
            field_qn = f"{object_qn}._field.{self._encode_segment(raw_key)}"
            field_depth = depth + 1
            is_scalar = not isinstance(raw_value, dict | list)
            value_str, value_type = self._classify_value(raw_value)
            self.ingestor.ensure_node_batch(
                cs.NodeLabel.JSON_FIELD,
                {
                    cs.KEY_QUALIFIED_NAME: field_qn,
                    cs.KEY_NAME: raw_key,
                    cs.KEY_PATH: file_path,
                    cs.KEY_JSON_KEY: raw_key,
                    cs.KEY_JSON_VALUE: value_str if is_scalar else None,
                    cs.KEY_JSON_VALUE_TYPE: value_type if is_scalar else None,
                    cs.KEY_JSON_DEPTH: field_depth,
                },
            )
            self.ingestor.ensure_relationship_batch(
                (cs.NodeLabel.JSON_OBJECT, cs.KEY_QUALIFIED_NAME, object_qn),
                cs.RelationshipType.HAS_FIELD,
                (cs.NodeLabel.JSON_FIELD, cs.KEY_QUALIFIED_NAME, field_qn),
            )
            value_label, value_qn = self._ingest_field_value(
                raw_value,
                field_qn,
                file_path,
                field_depth + 1,
            )
            self.ingestor.ensure_relationship_batch(
                (cs.NodeLabel.JSON_FIELD, cs.KEY_QUALIFIED_NAME, field_qn),
                cs.RelationshipType.HAS_VALUE,
                (value_label, cs.KEY_QUALIFIED_NAME, value_qn),
            )

        return cs.NodeLabel.JSON_OBJECT, object_qn

    def _ingest_array_node(
        self,
        array_qn: str,
        arr: list[Any],
        file_path: str,
        depth: int,
    ) -> tuple[cs.NodeLabel, str]:
        self.ingestor.ensure_node_batch(
            cs.NodeLabel.JSON_ARRAY,
            {
                cs.KEY_QUALIFIED_NAME: array_qn,
                cs.KEY_NAME: array_qn.split(cs.SEPARATOR_DOT)[-1],
                cs.KEY_PATH: file_path,
                cs.KEY_JSON_DEPTH: depth,
                cs.KEY_JSON_LENGTH: len(arr),
            },
        )

        for element_index, raw_value in enumerate(arr):
            value_label, value_qn = self._ingest_array_value(
                raw_value,
                array_qn,
                file_path,
                depth + 1,
                element_index,
            )
            self.ingestor.ensure_relationship_batch(
                (cs.NodeLabel.JSON_ARRAY, cs.KEY_QUALIFIED_NAME, array_qn),
                cs.RelationshipType.HAS_ELEMENT,
                (value_label, cs.KEY_QUALIFIED_NAME, value_qn),
                properties={cs.KEY_INDEX: element_index},
            )

        return cs.NodeLabel.JSON_ARRAY, array_qn

    def _ingest_field_value(
        self,
        value: Any,
        field_qn: str,
        file_path: str,
        depth: int,
    ) -> tuple[cs.NodeLabel, str]:
        if isinstance(value, dict):
            return self._ingest_object_node(f"{field_qn}._object", value, file_path, depth)
        if isinstance(value, list):
            return self._ingest_array_node(f"{field_qn}._array", value, file_path, depth)
        return self._ingest_scalar_node(f"{field_qn}._value", value, file_path, depth)

    def _ingest_array_value(
        self,
        value: Any,
        array_qn: str,
        file_path: str,
        depth: int,
        element_index: int,
    ) -> tuple[cs.NodeLabel, str]:
        element_base_qn = f"{array_qn}._index.{element_index}"
        if isinstance(value, dict):
            return self._ingest_object_node(
                f"{element_base_qn}._object",
                value,
                file_path,
                depth,
            )
        if isinstance(value, list):
            return self._ingest_array_node(
                f"{element_base_qn}._array",
                value,
                file_path,
                depth,
            )
        return self._ingest_scalar_node(
            f"{element_base_qn}._value",
            value,
            file_path,
            depth,
        )

    def _ingest_scalar_node(
        self,
        value_qn: str,
        value: Any,
        file_path: str,
        depth: int,
    ) -> tuple[cs.NodeLabel, str]:
        value_str, value_type = self._classify_value(value)
        name = value_str
        if len(name) > 50:
            name = name[:50] + "..."
        self.ingestor.ensure_node_batch(
            cs.NodeLabel.JSON_VALUE,
            {
                cs.KEY_QUALIFIED_NAME: value_qn,
                cs.KEY_NAME: name,
                cs.KEY_PATH: file_path,
                cs.KEY_JSON_VALUE: value_str,
                cs.KEY_JSON_VALUE_TYPE: value_type,
                cs.KEY_JSON_DEPTH: depth,
            },
        )
        return cs.NodeLabel.JSON_VALUE, value_qn

    def _build_base_qualified_name(self, relative_path: Path) -> str:
        path_parts = relative_path.with_suffix("").parts
        encoded_path_parts = [self._encode_segment(part) for part in path_parts]
        return cs.SEPARATOR_DOT.join(
            [self.project_name, "json", *encoded_path_parts]
        )

    @staticmethod
    def _encode_segment(segment: str) -> str:
        encoded = base64.urlsafe_b64encode(segment.encode(cs.ENCODING_UTF8)).decode(
            cs.ENCODING_UTF8
        )
        return f"s_{encoded.rstrip('=')}"

    @staticmethod
    def _classify_value(value: Any) -> tuple[str, str]:
        if value is None:
            return "null", "null"
        if isinstance(value, bool):
            return str(value).lower(), "boolean"
        if isinstance(value, int | float):
            return str(value), "number"
        return str(value), "string"
