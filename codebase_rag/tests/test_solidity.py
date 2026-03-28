"""Tests for Solidity language support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codebase_rag import constants as cs
from codebase_rag.constants import (
    LANGUAGE_METADATA,
    LanguageStatus,
    NodeLabel,
    RelationshipType,
    SupportedLanguage,
)
from codebase_rag.language_spec import LANGUAGE_FQN_SPECS, LANGUAGE_SPECS
from codebase_rag.parsers.handlers import get_handler
from codebase_rag.parsers.handlers.solidity import SolidityHandler
from codebase_rag.tests.conftest import create_mock_node
from codebase_rag.types_defs import NODE_SCHEMAS

if TYPE_CHECKING:
    from tree_sitter import Parser


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
        assert SupportedLanguage.SOLIDITY in LANGUAGE_FQN_SPECS

    def test_solidity_language_spec_in_dict(self) -> None:
        assert SupportedLanguage.SOLIDITY in LANGUAGE_SPECS

    def test_solidity_in_language_metadata(self) -> None:
        assert SupportedLanguage.SOLIDITY in LANGUAGE_METADATA
        metadata = LANGUAGE_METADATA[SupportedLanguage.SOLIDITY]
        assert metadata.status == LanguageStatus.FULL
        assert metadata.display_name == "Solidity"


class TestSolidityRelationshipTypes:
    """Tests for Solidity-specific relationship types."""

    def test_emits_relationship_exists(self) -> None:
        assert RelationshipType.EMITS == "EMITS"

    def test_modified_by_relationship_exists(self) -> None:
        assert RelationshipType.MODIFIED_BY == "MODIFIED_BY"

    def test_uses_library_relationship_exists(self) -> None:
        assert RelationshipType.USES_LIBRARY == "USES_LIBRARY"

    def test_calls_external_relationship_exists(self) -> None:
        assert RelationshipType.CALLS_EXTERNAL == "CALLS_EXTERNAL"

    def test_defines_event_relationship_exists(self) -> None:
        assert RelationshipType.DEFINES_EVENT == "DEFINES_EVENT"

    def test_defines_modifier_relationship_exists(self) -> None:
        assert RelationshipType.DEFINES_MODIFIER == "DEFINES_MODIFIER"

    def test_defines_state_relationship_exists(self) -> None:
        assert RelationshipType.DEFINES_STATE == "DEFINES_STATE"

    def test_calls_delegate_relationship_exists(self) -> None:
        assert RelationshipType.CALLS_DELEGATE == "CALLS_DELEGATE"

    def test_calls_static_relationship_exists(self) -> None:
        assert RelationshipType.CALLS_STATIC == "CALLS_STATIC"

    def test_reads_state_relationship_exists(self) -> None:
        assert RelationshipType.READS_STATE == "READS_STATE"

    def test_writes_state_relationship_exists(self) -> None:
        assert RelationshipType.WRITES_STATE == "WRITES_STATE"

    def test_reverts_with_relationship_exists(self) -> None:
        assert RelationshipType.REVERTS_WITH == "REVERTS_WITH"


class TestSolidityNodeSchemas:
    """Tests for Solidity NODE_SCHEMAS definitions."""

    def test_contract_schema_exists(self) -> None:
        contract_schemas = [s for s in NODE_SCHEMAS if s.label == NodeLabel.CONTRACT]
        assert len(contract_schemas) == 1
        schema = contract_schemas[0]
        assert "qualified_name" in schema.properties
        assert "name" in schema.properties
        assert "is_abstract" in schema.properties
        assert "path" in schema.properties
        assert "absolute_path" in schema.properties
        assert "start_line" in schema.properties
        assert "end_line" in schema.properties

    def test_library_schema_exists(self) -> None:
        library_schemas = [s for s in NODE_SCHEMAS if s.label == NodeLabel.LIBRARY]
        assert len(library_schemas) == 1
        schema = library_schemas[0]
        assert "qualified_name" in schema.properties
        assert "name" in schema.properties
        assert "path" in schema.properties
        assert "absolute_path" in schema.properties
        assert "start_line" in schema.properties
        assert "end_line" in schema.properties

    def test_event_schema_exists(self) -> None:
        event_schemas = [s for s in NODE_SCHEMAS if s.label == NodeLabel.EVENT]
        assert len(event_schemas) == 1
        schema = event_schemas[0]
        assert "qualified_name" in schema.properties
        assert "name" in schema.properties
        assert "parameters" in schema.properties
        assert "is_anonymous" in schema.properties
        assert "indexed_count" in schema.properties
        assert "path" in schema.properties
        assert "absolute_path" in schema.properties
        assert "start_line" in schema.properties
        assert "end_line" in schema.properties

    def test_modifier_schema_exists(self) -> None:
        modifier_schemas = [s for s in NODE_SCHEMAS if s.label == NodeLabel.MODIFIER]
        assert len(modifier_schemas) == 1
        schema = modifier_schemas[0]
        assert "qualified_name" in schema.properties
        assert "name" in schema.properties
        assert "parameters" in schema.properties
        assert "path" in schema.properties
        assert "absolute_path" in schema.properties
        assert "start_line" in schema.properties
        assert "end_line" in schema.properties

    def test_state_variable_schema_exists(self) -> None:
        state_var_schemas = [s for s in NODE_SCHEMAS if s.label == NodeLabel.STATE_VARIABLE]
        assert len(state_var_schemas) == 1
        schema = state_var_schemas[0]
        assert "qualified_name" in schema.properties
        assert "name" in schema.properties
        assert "type" in schema.properties
        assert "visibility" in schema.properties
        assert "is_constant" in schema.properties
        assert "is_immutable" in schema.properties
        assert "is_mapped" in schema.properties
        assert "path" in schema.properties
        assert "absolute_path" in schema.properties
        assert "start_line" in schema.properties
        assert "end_line" in schema.properties

    def test_custom_error_schema_exists(self) -> None:
        error_schemas = [s for s in NODE_SCHEMAS if s.label == NodeLabel.CUSTOM_ERROR]
        assert len(error_schemas) == 1
        schema = error_schemas[0]
        assert "qualified_name" in schema.properties
        assert "name" in schema.properties
        assert "parameters" in schema.properties
        assert "path" in schema.properties
        assert "absolute_path" in schema.properties
        assert "start_line" in schema.properties
        assert "end_line" in schema.properties


class TestSolidityHandlerBehavioral:
    """Behavioral tests for SolidityHandler methods."""

    def test_is_contract_function_inside_contract(self) -> None:
        handler = SolidityHandler()
        # Create a mock hierarchy: source_file -> contract_declaration -> function
        contract_node = create_mock_node(cs.TS_SOL_CONTRACT_DECLARATION)
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            parent=contract_node
        )

        assert handler.is_contract_function(func_node) is True

    def test_is_contract_function_outside_contract(self) -> None:
        handler = SolidityHandler()
        # Create a mock hierarchy: source_file -> function (no contract)
        source_file = create_mock_node(cs.TS_SOL_SOURCE_FILE)
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            parent=source_file
        )

        assert handler.is_contract_function(func_node) is False

    def test_is_contract_function_nested_deep(self) -> None:
        handler = SolidityHandler()
        # Create deep nesting: source_file -> contract -> some_node -> function
        contract_node = create_mock_node(cs.TS_SOL_CONTRACT_DECLARATION)
        intermediate = create_mock_node("block", parent=contract_node)
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            parent=intermediate
        )

        assert handler.is_contract_function(func_node) is True

    def test_is_class_method_same_as_is_contract_function(self) -> None:
        handler = SolidityHandler()
        # is_class_method should delegate to is_contract_function
        contract_node = create_mock_node(cs.TS_SOL_CONTRACT_DECLARATION)
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            parent=contract_node
        )

        assert handler.is_class_method(func_node) is True

    def test_extract_function_name_regular_function(self) -> None:
        handler = SolidityHandler()
        name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="myFunction")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            fields={cs.FIELD_NAME: name_node}
        )

        assert handler.extract_function_name(func_node) == "myFunction"

    def test_extract_function_name_constructor(self) -> None:
        handler = SolidityHandler()
        # Constructor nodes should return "constructor"
        constructor_node = create_mock_node(cs.TS_SOL_CONSTRUCTOR_DEFINITION)

        assert handler.extract_function_name(constructor_node) == "constructor"

    def test_extract_function_name_fallback_receive_payable(self) -> None:
        handler = SolidityHandler()
        # Payable fallback/receive should be "receive"
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="payable")
        fallback_node = create_mock_node(
            cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION,
            children=[mutability_node]
        )

        assert handler.extract_function_name(fallback_node) == "receive"

    def test_extract_function_name_fallback_receive_non_payable(self) -> None:
        handler = SolidityHandler()
        # Non-payable fallback/receive should be "fallback"
        fallback_node = create_mock_node(cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION)

        assert handler.extract_function_name(fallback_node) == "fallback"

    def test_extract_function_name_no_name_returns_none(self) -> None:
        handler = SolidityHandler()
        func_node = create_mock_node(cs.TS_SOL_FUNCTION_DEFINITION)

        assert handler.extract_function_name(func_node) is None

    def test_extract_visibility_present(self) -> None:
        handler = SolidityHandler()
        visibility_node = create_mock_node(cs.TS_SOL_VISIBILITY, text="public")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            children=[visibility_node]
        )

        assert handler.extract_visibility(func_node) == "public"

    def test_extract_visibility_not_present(self) -> None:
        handler = SolidityHandler()
        func_node = create_mock_node(cs.TS_SOL_FUNCTION_DEFINITION)

        assert handler.extract_visibility(func_node) is None

    def test_extract_state_mutability_view(self) -> None:
        handler = SolidityHandler()
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="view")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            children=[mutability_node]
        )

        assert handler.extract_state_mutability(func_node) == "view"

    def test_extract_state_mutability_not_present(self) -> None:
        handler = SolidityHandler()
        func_node = create_mock_node(cs.TS_SOL_FUNCTION_DEFINITION)

        assert handler.extract_state_mutability(func_node) is None

    def test_is_payable_true(self) -> None:
        handler = SolidityHandler()
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="payable")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            children=[mutability_node]
        )

        assert handler.is_payable(func_node) is True

    def test_is_payable_false(self) -> None:
        handler = SolidityHandler()
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="view")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            children=[mutability_node]
        )

        assert handler.is_payable(func_node) is False

    def test_is_view_or_pure_true_view(self) -> None:
        handler = SolidityHandler()
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="view")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            children=[mutability_node]
        )

        assert handler.is_view_or_pure(func_node) is True

    def test_is_view_or_pure_true_pure(self) -> None:
        handler = SolidityHandler()
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="pure")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            children=[mutability_node]
        )

        assert handler.is_view_or_pure(func_node) is True

    def test_is_view_or_pure_false(self) -> None:
        handler = SolidityHandler()
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="payable")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            children=[mutability_node]
        )

        assert handler.is_view_or_pure(func_node) is False

    def test_extract_inheritance_single_base(self) -> None:
        handler = SolidityHandler()
        # Create inheritance specifier with user_defined_type
        identifier_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="BaseContract")
        user_defined_type = create_mock_node(
            cs.TS_SOL_USER_DEFINED_TYPE,
            children=[identifier_node]
        )
        inheritance_specifier = create_mock_node(
            cs.TS_SOL_INHERITANCE_SPECIFIER,
            children=[user_defined_type]
        )
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            children=[inheritance_specifier]
        )

        result = handler.extract_inheritance(contract_node)
        assert result == ["BaseContract"]

    def test_extract_inheritance_multiple_bases(self) -> None:
        handler = SolidityHandler()
        # Create inheritance specifier with multiple user_defined_types
        id1 = create_mock_node(cs.TS_SOL_IDENTIFIER, text="Base1")
        udt1 = create_mock_node(cs.TS_SOL_USER_DEFINED_TYPE, children=[id1])
        id2 = create_mock_node(cs.TS_SOL_IDENTIFIER, text="Base2")
        udt2 = create_mock_node(cs.TS_SOL_USER_DEFINED_TYPE, children=[id2])
        inheritance_specifier = create_mock_node(
            cs.TS_SOL_INHERITANCE_SPECIFIER,
            children=[udt1, udt2]
        )
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            children=[inheritance_specifier]
        )

        result = handler.extract_inheritance(contract_node)
        assert result == ["Base1", "Base2"]

    def test_extract_inheritance_no_inheritance(self) -> None:
        handler = SolidityHandler()
        contract_node = create_mock_node(cs.TS_SOL_CONTRACT_DECLARATION)

        result = handler.extract_inheritance(contract_node)
        assert result == []

    def test_extract_base_class_name_user_defined_type(self) -> None:
        handler = SolidityHandler()
        identifier_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="BaseClass")
        user_defined_type = create_mock_node(
            cs.TS_SOL_USER_DEFINED_TYPE,
            children=[identifier_node]
        )

        assert handler.extract_base_class_name(user_defined_type) == "BaseClass"

    def test_extract_base_class_name_identifier(self) -> None:
        handler = SolidityHandler()
        identifier_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="SimpleBase")

        assert handler.extract_base_class_name(identifier_node) == "SimpleBase"

    def test_extract_base_class_name_no_text_returns_none(self) -> None:
        handler = SolidityHandler()
        empty_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="")

        assert handler.extract_base_class_name(empty_node) is None

    def test_extract_decorators_single_modifier(self) -> None:
        handler = SolidityHandler()
        # Create modifier invocation with identifier
        mod_identifier = create_mock_node(cs.TS_SOL_IDENTIFIER, text="onlyOwner")
        modifier_invocation = create_mock_node(
            cs.TS_SOL_MODIFIER_INVOCATION,
            children=[mod_identifier]
        )
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            children=[modifier_invocation]
        )

        result = handler.extract_decorators(func_node)
        assert result == ["onlyOwner"]

    def test_extract_decorators_multiple_modifiers(self) -> None:
        handler = SolidityHandler()
        mod_id1 = create_mock_node(cs.TS_SOL_IDENTIFIER, text="onlyOwner")
        mod_inv1 = create_mock_node(cs.TS_SOL_MODIFIER_INVOCATION, children=[mod_id1])
        mod_id2 = create_mock_node(cs.TS_SOL_IDENTIFIER, text="whenNotPaused")
        mod_inv2 = create_mock_node(cs.TS_SOL_MODIFIER_INVOCATION, children=[mod_id2])
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            children=[mod_inv1, mod_inv2]
        )

        result = handler.extract_decorators(func_node)
        assert result == ["onlyOwner", "whenNotPaused"]

    def test_extract_decorators_no_modifiers(self) -> None:
        handler = SolidityHandler()
        func_node = create_mock_node(cs.TS_SOL_FUNCTION_DEFINITION)

        result = handler.extract_decorators(func_node)
        assert result == []


# Integration tests with real tree-sitter parsing
try:
    import tree_sitter_solidity as tss

    SOLIDITY_PARSER_AVAILABLE = True
except ImportError:
    SOLIDITY_PARSER_AVAILABLE = False


@pytest.fixture
def solidity_parser() -> Parser | None:
    if not SOLIDITY_PARSER_AVAILABLE:
        return None
    from tree_sitter import Language, Parser

    language = Language(tss.language())
    return Parser(language)


@pytest.mark.skipif(not SOLIDITY_PARSER_AVAILABLE, reason="tree-sitter-solidity not available")
class TestSolidityHandlerWithRealParser:
    """Integration tests using real tree-sitter parsing."""

    def test_is_contract_function_real_parsing(self, solidity_parser: Parser) -> None:
        handler = SolidityHandler()
        code = b"""
contract MyContract {
    function myMethod() public returns (uint) {
        return 42;
    }
}
"""
        tree = solidity_parser.parse(code)
        # Find the function definition inside the contract
        contract_node = tree.root_node.children[0]
        contract_body = contract_node.child_by_field_name("body")
        func_node = contract_body.children[1]  # Skip '{' child

        assert handler.is_contract_function(func_node) is True

    def test_extract_function_name_real_parsing(self, solidity_parser: Parser) -> None:
        handler = SolidityHandler()
        code = b"""
contract MyContract {
    function getValue() public view returns (uint) {
        return value;
    }
}
"""
        tree = solidity_parser.parse(code)
        contract_node = tree.root_node.children[0]
        contract_body = contract_node.child_by_field_name("body")
        func_node = contract_body.children[1]

        assert handler.extract_function_name(func_node) == "getValue"

    def test_extract_function_name_constructor_real(self, solidity_parser: Parser) -> None:
        handler = SolidityHandler()
        code = b"""
contract MyContract {
    constructor() public {
        owner = msg.sender;
    }
}
"""
        tree = solidity_parser.parse(code)
        contract_node = tree.root_node.children[0]
        contract_body = contract_node.child_by_field_name("body")

        # Find constructor node
        constructor_node = None
        for child in contract_body.children:
            if child.type == cs.TS_SOL_CONSTRUCTOR_DEFINITION:
                constructor_node = child
                break

        assert constructor_node is not None
        assert handler.extract_function_name(constructor_node) == "constructor"

    def test_extract_visibility_real_parsing(self, solidity_parser: Parser) -> None:
        handler = SolidityHandler()
        code = b"""
contract MyContract {
    function publicMethod() public returns (uint) {
        return 1;
    }
    function privateMethod() private returns (uint) {
        return 2;
    }
}
"""
        tree = solidity_parser.parse(code)
        contract_node = tree.root_node.children[0]
        contract_body = contract_node.child_by_field_name("body")

        # Find public function node
        public_func = None
        for child in contract_body.children:
            if child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode()
                    if name == "publicMethod":
                        public_func = child
                    elif name == "privateMethod":
                        _private_func = child  # noqa: F841

        assert public_func is not None
        result = handler.extract_visibility(public_func)
        assert result == "public"

    def test_extract_state_mutability_real_parsing(self, solidity_parser: Parser) -> None:
        handler = SolidityHandler()
        code = b"""
contract MyContract {
    function viewMethod() public view returns (uint) {
        return value;
    }
    function pureMethod() public pure returns (uint) {
        return 42;
    }
}
"""
        tree = solidity_parser.parse(code)
        contract_node = tree.root_node.children[0]
        contract_body = contract_node.child_by_field_name("body")

        view_func = None
        for child in contract_body.children:
            if child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode()
                    if name == "viewMethod":
                        view_func = child
                    elif name == "pureMethod":
                        _pure_func = child  # noqa: F841

        assert view_func is not None
        result = handler.extract_state_mutability(view_func)
        assert result == "view"

    def test_extract_inheritance_real_parsing(self, solidity_parser: Parser) -> None:
        handler = SolidityHandler()
        code = b"""
contract ChildContract is BaseContract {
    function myMethod() public returns (uint) {
        return 42;
    }
}
"""
        tree = solidity_parser.parse(code)
        contract_node = tree.root_node.children[0]

        result = handler.extract_inheritance(contract_node)
        assert result == ["BaseContract"]

    def test_extract_inheritance_multiple_real(self, solidity_parser: Parser) -> None:
        handler = SolidityHandler()
        code = b"""
contract ChildContract is BaseContract, AnotherBase {
    function myMethod() public returns (uint) {
        return 42;
    }
}
"""
        tree = solidity_parser.parse(code)
        contract_node = tree.root_node.children[0]

        result = handler.extract_inheritance(contract_node)
        assert "BaseContract" in result
        assert "AnotherBase" in result

    def test_extract_decorators_real_parsing(self, solidity_parser: Parser) -> None:
        handler = SolidityHandler()
        code = b"""
contract MyContract {
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    function restrictedMethod() public onlyOwner returns (uint) {
        return 42;
    }
}
"""
        tree = solidity_parser.parse(code)
        contract_node = tree.root_node.children[0]
        contract_body = contract_node.child_by_field_name("body")

        # Find the function with modifier
        func_node = None
        for child in contract_body.children:
            if child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                name_node = child.child_by_field_name("name")
                if name_node and name_node.text.decode() == "restrictedMethod":
                    func_node = child
                    break

        assert func_node is not None
        result = handler.extract_decorators(func_node)
        assert "onlyOwner" in result
