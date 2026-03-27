"""Tests for Solidity language support."""
from __future__ import annotations

import pytest
from pathlib import Path

from codebase_rag.constants import NodeLabel, SupportedLanguage
from codebase_rag.parsers.handlers import get_handler
from codebase_rag.parsers.handlers.solidity import SolidityHandler


class TestSolidityHandler:
    """Tests for SolidityHandler."""

    def test_returns_solidity_handler_for_solidity(self) -> None:
        handler = get_handler(SupportedLanguage.SOLIDITY)
        assert isinstance(handler, SolidityHandler)

    def test_solidity_handler_extends_base(self) -> None:
        assert issubclass(SolidityHandler, get_handler(SupportedLanguage.PYTHON).__class__.__bases__[0])

    def test_handler_has_all_protocol_methods(self) -> None:
        handler = get_handler(SupportedLanguage.SOLIDITY)
        assert hasattr(handler, "is_inside_method_with_object_literals")
        assert hasattr(handler, "is_class_method")
        assert hasattr(handler, "is_export_inside_function")
        assert hasattr(handler, "extract_function_name")
        assert hasattr(handler, "build_function_qualified_name")
        assert hasattr(handler, "is_function_exported")
        assert hasattr(handler, "should_process_as_impl_block")
        assert hasattr(handler, "extract_impl_target")
        assert hasattr(handler, "build_method_qualified_name")
        assert hasattr(handler, "extract_base_class_name")
        assert hasattr(handler, "build_nested_function_qn")
        assert hasattr(handler, "extract_decorators")

    def test_handler_methods_are_callable(self) -> None:
        handler = get_handler(SupportedLanguage.SOLIDITY)
        assert callable(handler.is_inside_method_with_object_literals)
        assert callable(handler.is_class_method)
        assert callable(handler.is_export_inside_function)
        assert callable(handler.extract_function_name)
        assert callable(handler.build_function_qualified_name)
        assert callable(handler.is_function_exported)
        assert callable(handler.should_process_as_impl_block)
        assert callable(handler.extract_impl_target)
        assert callable(handler.build_method_qualified_name)
        assert callable(handler.extract_base_class_name)
        assert callable(handler.build_nested_function_qn)
        assert callable(handler.extract_decorators)

    def test_same_instance_returned_for_same_language(self) -> None:
        handler1 = get_handler(SupportedLanguage.SOLIDITY)
        handler2 = get_handler(SupportedLanguage.SOLIDITY)
        assert handler1 is handler2

    def test_different_instances_for_different_languages(self) -> None:
        sol_handler = get_handler(SupportedLanguage.SOLIDITY)
        py_handler = get_handler(SupportedLanguage.PYTHON)
        assert sol_handler is not py_handler


class TestSolidityHandlerSpecificMethods:
    """Tests for Solidity-specific handler methods."""

    def test_is_contract_function(self) -> None:
        handler = SolidityHandler()
        # Test that the method exists and is callable
        assert callable(handler.is_contract_function)

    def test_extract_visibility(self) -> None:
        handler = SolidityHandler()
        assert callable(handler.extract_visibility)

    def test_extract_state_mutability(self) -> None:
        handler = SolidityHandler()
        assert callable(handler.extract_state_mutability)

    def test_is_payable(self) -> None:
        handler = SolidityHandler()
        assert callable(handler.is_payable)

    def test_is_view_or_pure(self) -> None:
        handler = SolidityHandler()
        assert callable(handler.is_view_or_pure)

    def test_extract_inheritance(self) -> None:
        handler = SolidityHandler()
        assert callable(handler.extract_inheritance)


class TestSolidityNodeLabels:
    """Tests for Solidity-specific node labels."""

    def test_contract_label_exists(self) -> None:
        assert NodeLabel.CONTRACT == "Contract"

    def test_library_label_exists(self) -> None:
        assert NodeLabel.LIBRARY == "Library"

    def test_event_label_exists(self) -> None:
        assert NodeLabel.EVENT == "Event"

    def test_modifier_label_exists(self) -> None:
        assert NodeLabel.MODIFIER == "Modifier"

    def test_state_variable_label_exists(self) -> None:
        assert NodeLabel.STATE_VARIABLE == "StateVariable"

    def test_custom_error_label_exists(self) -> None:
        assert NodeLabel.CUSTOM_ERROR == "CustomError"


class TestSolidityLanguageSpec:
    """Tests for Solidity language specification."""

    def test_solidity_language_in_enum(self) -> None:
        assert SupportedLanguage.SOLIDITY == "solidity"

    def test_solidity_extensions_defined(self) -> None:
        from codebase_rag import constants as cs
        assert hasattr(cs, "SOLIDITY_EXTENSIONS")
        assert cs.SOLIDITY_EXTENSIONS == (".sol",)

    def test_solidity_tree_sitter_module_defined(self) -> None:
        from codebase_rag import constants as cs
        assert hasattr(cs, "TreeSitterModule")
        assert cs.TreeSitterModule.SOLIDITY == "tree_sitter_solidity"

    def test_solidity_node_types_defined(self) -> None:
        from codebase_rag import constants as cs
        assert hasattr(cs, "TS_SOL_SOURCE_FILE")
        assert hasattr(cs, "TS_SOL_CONTRACT_DECLARATION")
        assert hasattr(cs, "TS_SOL_INTERFACE_DECLARATION")
        assert hasattr(cs, "TS_SOL_LIBRARY_DECLARATION")
        assert hasattr(cs, "TS_SOL_FUNCTION_DEFINITION")
        assert hasattr(cs, "TS_SOL_MODIFIER_DEFINITION")
        assert hasattr(cs, "TS_SOL_EVENT_DEFINITION")
        assert hasattr(cs, "TS_SOL_STRUCT_DECLARATION")
        assert hasattr(cs, "TS_SOL_ENUM_DECLARATION")
        assert hasattr(cs, "TS_SOL_ERROR_DECLARATION")
        assert hasattr(cs, "TS_SOL_STATE_VARIABLE_DECLARATION")
        assert hasattr(cs, "TS_SOL_CALL_EXPRESSION")
        assert hasattr(cs, "TS_SOL_MEMBER_EXPRESSION")
        assert hasattr(cs, "TS_SOL_IMPORT_DIRECTIVE")
        assert hasattr(cs, "TS_SOL_EMIT_STATEMENT")
        assert hasattr(cs, "TS_SOL_CONSTRUCTOR_DEFINITION")
        assert hasattr(cs, "TS_SOL_FALLBACK_RECEIVE_DEFINITION")

    def test_solidity_fqn_spec_in_dict(self) -> None:
        from codebase_rag.language_spec import LANGUAGE_FQN_SPECS
        from codebase_rag.constants import SupportedLanguage
        assert SupportedLanguage.SOLIDITY in LANGUAGE_FQN_SPECS

    def test_solidity_language_spec_in_dict(self) -> None:
        from codebase_rag.language_spec import LANGUAGE_SPECS
        from codebase_rag.constants import SupportedLanguage
        assert SupportedLanguage.SOLIDITY in LANGUAGE_SPECS

    def test_solidity_in_language_metadata(self) -> None:
        from codebase_rag.constants import LANGUAGE_METADATA, SupportedLanguage, LanguageStatus
        assert SupportedLanguage.SOLIDITY in LANGUAGE_METADATA
        metadata = LANGUAGE_METADATA[SupportedLanguage.SOLIDITY]
        assert metadata.status == LanguageStatus.FULL
        assert metadata.display_name == "Solidity"


class TestSolidityRelationshipTypes:
    """Tests for Solidity-specific relationship types."""

    def test_emits_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.EMITS == "EMITS"

    def test_modified_by_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.MODIFIED_BY == "MODIFIED_BY"

    def test_uses_library_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.USES_LIBRARY == "USES_LIBRARY"

    def test_calls_external_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.CALLS_EXTERNAL == "CALLS_EXTERNAL"

    def test_defines_event_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.DEFINES_EVENT == "DEFINES_EVENT"

    def test_defines_modifier_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.DEFINES_MODIFIER == "DEFINES_MODIFIER"

    def test_defines_state_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.DEFINES_STATE == "DEFINES_STATE"

    def test_calls_delegate_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.CALLS_DELEGATE == "CALLS_DELEGATE"

    def test_calls_static_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.CALLS_STATIC == "CALLS_STATIC"

    def test_reads_state_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.READS_STATE == "READS_STATE"

    def test_writes_state_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.WRITES_STATE == "WRITES_STATE"

    def test_reverts_with_relationship_exists(self) -> None:
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.REVERTS_WITH == "REVERTS_WITH"