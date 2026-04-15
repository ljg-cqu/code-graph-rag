from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from tree_sitter import Node

from ... import constants as cs
from ... import logs
from ...types_defs import NodeType, PropertyDict
from ..utils import safe_decode_text

if TYPE_CHECKING:
    from ...services import IngestorProtocol


def _extract_name(node: Node) -> str | None:
    name_node = node.child_by_field_name(cs.FIELD_NAME)
    if name_node:
        return safe_decode_text(name_node)
    for child in node.children:
        if child.type == cs.TS_SOL_IDENTIFIER:
            return safe_decode_text(child)
    return None


def _extract_parameters_text(node: Node) -> list[str]:
    params_node = node.child_by_field_name(cs.FIELD_PARAMETERS)
    if not params_node:
        return []
    text = safe_decode_text(params_node)
    if text is None:
        return []
    stripped = text.strip("()")
    if not stripped.strip():
        return []
    return [p.strip() for p in stripped.split(",") if p.strip()]


def _set_path_props(props: PropertyDict, file_path: Path, repo_path: Path) -> None:
    props[cs.KEY_PATH] = file_path.relative_to(repo_path).as_posix()
    props[cs.KEY_ABSOLUTE_PATH] = file_path.resolve().as_posix()


def _ingest_event(
    node: Node,
    contract_qn: str,
    container_label: cs.NodeLabel,
    ingestor: IngestorProtocol,
    file_path: Path | None,
    repo_path: Path,
) -> None:
    name = _extract_name(node)
    if not name:
        return

    event_qn = f"{contract_qn}.{name}"
    parameters = _extract_parameters_text(node)

    indexed_count = 0
    is_anonymous = False
    for child in node.children:
        if child.type == "anonymous":
            is_anonymous = True
        if child.type in ("event_param_list", "event_parameter_list"):
            for param in child.children:
                if safe_decode_text(param) == "indexed":
                    indexed_count += 1

    props: PropertyDict = {
        cs.KEY_QUALIFIED_NAME: event_qn,
        cs.KEY_NAME: name,
        cs.KEY_PARAMETERS: parameters,
        cs.KEY_IS_ANONYMOUS: is_anonymous,
        cs.KEY_INDEXED_COUNT: indexed_count,
        cs.KEY_START_LINE: node.start_point[0] + 1,
        cs.KEY_END_LINE: node.end_point[0] + 1,
    }
    if file_path is not None:
        _set_path_props(props, file_path, repo_path)

    logger.info(logs.SOL_FOUND_EVENT.format(name=name, qn=event_qn))
    ingestor.ensure_node_batch(cs.NodeLabel.EVENT, props)
    ingestor.ensure_relationship_batch(
        (container_label, cs.KEY_QUALIFIED_NAME, contract_qn),
        cs.RelationshipType.DEFINES_EVENT,
        (cs.NodeLabel.EVENT, cs.KEY_QUALIFIED_NAME, event_qn),
    )


def _ingest_modifier(
    node: Node,
    contract_qn: str,
    ingestor: IngestorProtocol,
    file_path: Path | None,
    repo_path: Path,
) -> None:
    name = _extract_name(node)
    if not name:
        return

    modifier_qn = f"{contract_qn}.{name}"
    parameters = _extract_parameters_text(node)

    props: PropertyDict = {
        cs.KEY_QUALIFIED_NAME: modifier_qn,
        cs.KEY_NAME: name,
        cs.KEY_PARAMETERS: parameters,
        cs.KEY_START_LINE: node.start_point[0] + 1,
        cs.KEY_END_LINE: node.end_point[0] + 1,
    }
    if file_path is not None:
        _set_path_props(props, file_path, repo_path)

    logger.info(logs.SOL_FOUND_MODIFIER.format(name=name, qn=modifier_qn))
    ingestor.ensure_node_batch(cs.NodeLabel.MODIFIER, props)
    ingestor.ensure_relationship_batch(
        (cs.NodeLabel.CONTRACT, cs.KEY_QUALIFIED_NAME, contract_qn),
        cs.RelationshipType.DEFINES_MODIFIER,
        (cs.NodeLabel.MODIFIER, cs.KEY_QUALIFIED_NAME, modifier_qn),
    )


def _ingest_state_variable(
    node: Node,
    contract_qn: str,
    container_label: cs.NodeLabel,
    ingestor: IngestorProtocol,
    file_path: Path | None,
    repo_path: Path,
) -> None:
    name = _extract_name(node)
    if not name:
        return

    state_var_qn = f"{contract_qn}.{name}"

    type_node = node.child_by_field_name(cs.FIELD_TYPE)
    type_text = safe_decode_text(type_node) or ""

    visibility = "internal"
    is_constant = False
    is_immutable = False
    for child in node.children:
        if child.type == cs.TS_SOL_VISIBILITY:
            visibility = safe_decode_text(child) or "internal"
        elif child.type == "constant":
            is_constant = True
        elif child.type == "immutable":
            is_immutable = True

    is_mapped = "mapping" in type_text

    props: PropertyDict = {
        cs.KEY_QUALIFIED_NAME: state_var_qn,
        cs.KEY_NAME: name,
        cs.KEY_TYPE: type_text,
        cs.KEY_IS_CONSTANT: is_constant,
        cs.KEY_IS_IMMUTABLE: is_immutable,
        cs.KEY_IS_MAPPED: is_mapped,
        cs.KEY_START_LINE: node.start_point[0] + 1,
        cs.KEY_END_LINE: node.end_point[0] + 1,
    }
    if visibility:
        props["visibility"] = visibility
    if file_path is not None:
        _set_path_props(props, file_path, repo_path)

    logger.info(logs.SOL_FOUND_STATE_VAR.format(name=name, qn=state_var_qn))
    ingestor.ensure_node_batch(cs.NodeLabel.STATE_VARIABLE, props)
    ingestor.ensure_relationship_batch(
        (container_label, cs.KEY_QUALIFIED_NAME, contract_qn),
        cs.RelationshipType.DEFINES_STATE,
        (cs.NodeLabel.STATE_VARIABLE, cs.KEY_QUALIFIED_NAME, state_var_qn),
    )


def _ingest_custom_error(
    node: Node,
    contract_qn: str,
    container_label: cs.NodeLabel,
    ingestor: IngestorProtocol,
    file_path: Path | None,
    repo_path: Path,
) -> None:
    name = _extract_name(node)
    if not name:
        return

    error_qn = f"{contract_qn}.{name}"
    parameters = _extract_parameters_text(node)

    props: PropertyDict = {
        cs.KEY_QUALIFIED_NAME: error_qn,
        cs.KEY_NAME: name,
        cs.KEY_PARAMETERS: parameters,
        cs.KEY_START_LINE: node.start_point[0] + 1,
        cs.KEY_END_LINE: node.end_point[0] + 1,
    }
    if file_path is not None:
        _set_path_props(props, file_path, repo_path)

    logger.info(logs.SOL_FOUND_CUSTOM_ERROR.format(name=name, qn=error_qn))
    ingestor.ensure_node_batch(cs.NodeLabel.CUSTOM_ERROR, props)
    ingestor.ensure_relationship_batch(
        (container_label, cs.KEY_QUALIFIED_NAME, contract_qn),
        cs.RelationshipType.DEFINES_CUSTOM_ERROR,
        (cs.NodeLabel.CUSTOM_ERROR, cs.KEY_QUALIFIED_NAME, error_qn),
    )


def ingest_solidity_contract_members(
    contract_node: Node,
    contract_qn: str,
    contract_type: NodeType,
    ingestor: IngestorProtocol,
    file_path: Path | None,
    repo_path: Path,
) -> None:
    body_node = contract_node.child_by_field_name("body")
    if not body_node:
        return

    container_label = cs.NodeLabel(contract_type)
    for child in body_node.children:
        match child.type:
            case cs.TS_SOL_EVENT_DEFINITION:
                _ingest_event(
                    child, contract_qn, container_label, ingestor, file_path, repo_path
                )
            case cs.TS_SOL_MODIFIER_DEFINITION if contract_type == NodeType.CONTRACT:
                _ingest_modifier(child, contract_qn, ingestor, file_path, repo_path)
            case cs.TS_SOL_STATE_VARIABLE_DECLARATION:
                _ingest_state_variable(
                    child, contract_qn, container_label, ingestor, file_path, repo_path
                )
            case cs.TS_SOL_ERROR_DECLARATION:
                _ingest_custom_error(
                    child, contract_qn, container_label, ingestor, file_path, repo_path
                )
