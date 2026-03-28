"""Tests for Solidity Fully Qualified Name (FQN) generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codebase_rag import constants as cs
from codebase_rag.language_spec import (
    SOLIDITY_FQN_SPEC,
    _solidity_get_name,
    _solidity_file_to_module,
)
from codebase_rag.utils.fqn_resolver import resolve_fqn_from_ast

if TYPE_CHECKING:
    from tree_sitter import Parser

from codebase_rag.tests.conftest import create_mock_node

# Check if tree-sitter-solidity is available
try:
    import tree_sitter_solidity as tss

    SOLIDITY_PARSER_AVAILABLE = True
except ImportError:
    SOLIDITY_PARSER_AVAILABLE = False


@pytest.fixture
def solidity_parser() -> "Parser | None":
    if not SOLIDITY_PARSER_AVAILABLE:
        return None
    from tree_sitter import Language, Parser

    language = Language(tss.language())
    return Parser(language)


class TestSolidityGetName:
    """Tests for _solidity_get_name function."""

    def test_contract_name(self) -> None:
        """Extract name from contract declaration."""
        name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="MyContract")
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            fields={cs.FIELD_NAME: name_node}
        )
        result = _solidity_get_name(contract_node)
        assert result == "MyContract"

    def test_interface_name(self) -> None:
        """Extract name from interface declaration."""
        name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="IMyInterface")
        interface_node = create_mock_node(
            cs.TS_SOL_INTERFACE_DECLARATION,
            fields={cs.FIELD_NAME: name_node}
        )
        result = _solidity_get_name(interface_node)
        assert result == "IMyInterface"

    def test_library_name(self) -> None:
        """Extract name from library declaration."""
        name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="MyLibrary")
        library_node = create_mock_node(
            cs.TS_SOL_LIBRARY_DECLARATION,
            fields={cs.FIELD_NAME: name_node}
        )
        result = _solidity_get_name(library_node)
        assert result == "MyLibrary"

    def test_function_name(self) -> None:
        """Extract name from function definition."""
        name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="myFunction")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            fields={cs.FIELD_NAME: name_node}
        )
        result = _solidity_get_name(func_node)
        assert result == "myFunction"

    def test_constructor_name(self) -> None:
        """Constructor should return 'constructor' as name."""
        constructor_node = create_mock_node(cs.TS_SOL_CONSTRUCTOR_DEFINITION)
        result = _solidity_get_name(constructor_node)
        assert result == "constructor"

    def test_receive_name(self) -> None:
        """Receive function (payable) should return 'receive' as name."""
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="payable")
        receive_node = create_mock_node(
            cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION,
            children=[mutability_node]
        )
        result = _solidity_get_name(receive_node)
        assert result == "receive"

    def test_fallback_name(self) -> None:
        """Fallback function (non-payable) should return 'fallback' as name."""
        # No payable modifier = fallback
        fallback_node = create_mock_node(cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION)
        result = _solidity_get_name(fallback_node)
        assert result == "fallback"

    def test_fallback_with_non_payable_mutability(self) -> None:
        """Fallback with non-payable mutability should return 'fallback'."""
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="nonpayable")
        fallback_node = create_mock_node(
            cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION,
            children=[mutability_node]
        )
        result = _solidity_get_name(fallback_node)
        assert result == "fallback"

    def test_modifier_name(self) -> None:
        """Extract name from modifier definition."""
        name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="onlyOwner")
        modifier_node = create_mock_node(
            cs.TS_SOL_MODIFIER_DEFINITION,
            fields={cs.FIELD_NAME: name_node}
        )
        result = _solidity_get_name(modifier_node)
        assert result == "onlyOwner"

    def test_no_name_returns_none(self) -> None:
        """Function without name field should return None."""
        func_node = create_mock_node(cs.TS_SOL_FUNCTION_DEFINITION)
        result = _solidity_get_name(func_node)
        assert result is None


class TestSolidityFileToModule:
    """Tests for _solidity_file_to_module function."""

    def test_basic_path(self) -> None:
        """Basic path without prefix stripping."""
        file_path = Path("/repo/src/contracts/MyContract.sol")
        repo_root = Path("/repo")
        result = _solidity_file_to_module(file_path, repo_root)
        assert result == ["contracts", "MyContract"]

    def test_src_prefix_stripped(self) -> None:
        """src/ prefix should be stripped."""
        file_path = Path("/repo/src/MyContract.sol")
        repo_root = Path("/repo")
        result = _solidity_file_to_module(file_path, repo_root)
        assert result == ["MyContract"]

    def test_contracts_prefix_stripped(self) -> None:
        """contracts/ prefix should be stripped."""
        file_path = Path("/repo/contracts/MyContract.sol")
        repo_root = Path("/repo")
        result = _solidity_file_to_module(file_path, repo_root)
        assert result == ["MyContract"]

    def test_script_prefix_stripped(self) -> None:
        """script/ prefix should be stripped."""
        file_path = Path("/repo/script/Deploy.s.sol")
        repo_root = Path("/repo")
        result = _solidity_file_to_module(file_path, repo_root)
        assert result == ["Deploy.s"]

    def test_test_prefix_stripped(self) -> None:
        """test/ prefix should be stripped."""
        file_path = Path("/repo/test/MyContract.t.sol")
        repo_root = Path("/repo")
        result = _solidity_file_to_module(file_path, repo_root)
        assert result == ["MyContract.t"]

    def test_nested_path(self) -> None:
        """Nested path should preserve subdirectories after prefix."""
        file_path = Path("/repo/src/tokens/ERC20.sol")
        repo_root = Path("/repo")
        result = _solidity_file_to_module(file_path, repo_root)
        assert result == ["tokens", "ERC20"]

    def test_no_prefix_preserved(self) -> None:
        """Path without common prefix should be fully preserved."""
        file_path = Path("/repo/lib/MyContract.sol")
        repo_root = Path("/repo")
        result = _solidity_file_to_module(file_path, repo_root)
        assert result == ["lib", "MyContract"]


class TestSolidityFQNSpec:
    """Tests for SOLIDITY_FQN_SPEC configuration."""

    def test_scope_node_types(self) -> None:
        """Verify scope node types are correctly defined."""
        expected_scope_types = frozenset({
            cs.TS_SOL_CONTRACT_DECLARATION,
            cs.TS_SOL_INTERFACE_DECLARATION,
            cs.TS_SOL_LIBRARY_DECLARATION,
            cs.TS_SOL_SOURCE_FILE,
        })
        assert SOLIDITY_FQN_SPEC.scope_node_types == expected_scope_types

    def test_function_node_types(self) -> None:
        """Verify function node types are correctly defined."""
        expected_function_types = frozenset({
            cs.TS_SOL_FUNCTION_DEFINITION,
            cs.TS_SOL_MODIFIER_DEFINITION,
            cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION,
            cs.TS_SOL_CONSTRUCTOR_DEFINITION,
        })
        assert SOLIDITY_FQN_SPEC.function_node_types == expected_function_types

    def test_get_name_is_solidity_get_name(self) -> None:
        """Verify get_name function is _solidity_get_name."""
        assert SOLIDITY_FQN_SPEC.get_name == _solidity_get_name

    def test_file_to_module_parts_is_solidity_file_to_module(self) -> None:
        """Verify file_to_module_parts is _solidity_file_to_module."""
        assert SOLIDITY_FQN_SPEC.file_to_module_parts == _solidity_file_to_module


class TestSolidityFQNGeneration:
    """Tests for FQN generation using resolve_fqn_from_ast with mock nodes.

    FQN Format: {project}.{module_parts}.{scope_name}.{function_name}

    Where:
    - module_parts: file path parts after stripping common prefixes (src/, contracts/, etc.)
    - scope_name: contract/interface/library name (if function is inside one)
    - function_name: the actual function name
    """

    def test_contract_fqn(self) -> None:
        """Contract FQN should be: project.module.ContractName.

        Note: While resolve_fqn_from_ast is primarily designed for function nodes,
        it can technically resolve contract FQNs since contracts also have names
        and are in scope_node_types. The FQN for a contract is:
        project.module.ContractName (where the contract name appears twice:
        once from the file path and once as the scope name).
        """
        # Build hierarchy: source_file -> contract
        name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="MyContract")
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            fields={cs.FIELD_NAME: name_node}
        )
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[contract_node]
        )
        contract_node.node_parent = source_file

        file_path = Path("/repo/src/MyContract.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            contract_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        # Contract FQN can be resolved (though primarily used for functions)
        # The contract name appears in both module path and as scope name
        assert result == "myproject.MyContract.MyContract"

    def test_function_inside_contract_fqn(self) -> None:
        """Function FQN inside contract should be: project.module.ContractName.functionName.

        Example: myproject.MyContract.MyContract.getValue
        - module_parts: ["MyContract"] (from src/MyContract.sol)
        - scope_name: "MyContract" (the contract)
        - function_name: "getValue"
        """
        # Build hierarchy: source_file -> contract -> function
        func_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="getValue")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            fields={cs.FIELD_NAME: func_name_node}
        )
        contract_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="MyContract")
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            fields={cs.FIELD_NAME: contract_name_node},
            children=[func_node]
        )
        func_node.node_parent = contract_node
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[contract_node]
        )
        contract_node.node_parent = source_file

        file_path = Path("/repo/src/MyContract.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert result == "myproject.MyContract.MyContract.getValue"

    def test_library_function_fqn(self) -> None:
        """Function inside library should have FQN: project.module.LibraryName.functionName."""
        func_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="safeAdd")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            fields={cs.FIELD_NAME: func_name_node}
        )
        lib_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="SafeMath")
        library_node = create_mock_node(
            cs.TS_SOL_LIBRARY_DECLARATION,
            fields={cs.FIELD_NAME: lib_name_node},
            children=[func_node]
        )
        func_node.node_parent = library_node
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[library_node]
        )
        library_node.node_parent = source_file

        file_path = Path("/repo/src/libraries/SafeMath.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert result == "myproject.libraries.SafeMath.SafeMath.safeAdd"

    def test_interface_function_fqn(self) -> None:
        """Function inside interface should have FQN: project.module.InterfaceName.functionName."""
        func_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="transfer")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            fields={cs.FIELD_NAME: func_name_node}
        )
        interface_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="IERC20")
        interface_node = create_mock_node(
            cs.TS_SOL_INTERFACE_DECLARATION,
            fields={cs.FIELD_NAME: interface_name_node},
            children=[func_node]
        )
        func_node.node_parent = interface_node
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[interface_node]
        )
        interface_node.node_parent = source_file

        file_path = Path("/repo/src/interfaces/IERC20.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert result == "myproject.interfaces.IERC20.IERC20.transfer"

    def test_constructor_fqn(self) -> None:
        """Constructor FQN should be: project.module.ContractName.constructor."""
        constructor_node = create_mock_node(cs.TS_SOL_CONSTRUCTOR_DEFINITION)
        contract_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="MyContract")
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            fields={cs.FIELD_NAME: contract_name_node},
            children=[constructor_node]
        )
        constructor_node.node_parent = contract_node
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[contract_node]
        )
        contract_node.node_parent = source_file

        file_path = Path("/repo/src/MyContract.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            constructor_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert result == "myproject.MyContract.MyContract.constructor"

    def test_receive_fqn(self) -> None:
        """Receive function FQN should be: project.module.ContractName.receive."""
        mutability_node = create_mock_node(cs.TS_SOL_STATE_MUTABILITY, text="payable")
        receive_node = create_mock_node(
            cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION,
            children=[mutability_node]
        )
        contract_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="MyContract")
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            fields={cs.FIELD_NAME: contract_name_node},
            children=[receive_node]
        )
        receive_node.node_parent = contract_node
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[contract_node]
        )
        contract_node.node_parent = source_file

        file_path = Path("/repo/src/MyContract.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            receive_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert result == "myproject.MyContract.MyContract.receive"

    def test_fallback_fqn(self) -> None:
        """Fallback function FQN should be: project.module.ContractName.fallback."""
        # No payable modifier = fallback
        fallback_node = create_mock_node(cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION)
        contract_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="MyContract")
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            fields={cs.FIELD_NAME: contract_name_node},
            children=[fallback_node]
        )
        fallback_node.node_parent = contract_node
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[contract_node]
        )
        contract_node.node_parent = source_file

        file_path = Path("/repo/src/MyContract.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            fallback_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert result == "myproject.MyContract.MyContract.fallback"

    def test_modifier_fqn(self) -> None:
        """Modifier FQN inside contract should be: project.module.ContractName.modifierName."""
        modifier_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="onlyOwner")
        modifier_node = create_mock_node(
            cs.TS_SOL_MODIFIER_DEFINITION,
            fields={cs.FIELD_NAME: modifier_name_node}
        )
        contract_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="MyContract")
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            fields={cs.FIELD_NAME: contract_name_node},
            children=[modifier_node]
        )
        modifier_node.node_parent = contract_node
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[contract_node]
        )
        contract_node.node_parent = source_file

        file_path = Path("/repo/src/MyContract.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            modifier_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert result == "myproject.MyContract.MyContract.onlyOwner"

    def test_free_function_fqn(self) -> None:
        """Free function (outside contract) FQN should be: project.module.functionName."""
        func_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="helper")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            fields={cs.FIELD_NAME: func_name_node}
        )
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[func_node]
        )
        func_node.node_parent = source_file

        file_path = Path("/repo/src/utils.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        # source_file is in scope_node_types but has no name, so it's not added to path
        # Expected: myproject.utils.helper
        assert result == "myproject.utils.helper"

    def test_nested_contract_path(self) -> None:
        """Nested contract path should include subdirectories in module parts."""
        func_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="mint")
        func_node = create_mock_node(
            cs.TS_SOL_FUNCTION_DEFINITION,
            fields={cs.FIELD_NAME: func_name_node}
        )
        contract_name_node = create_mock_node(cs.TS_SOL_IDENTIFIER, text="MyToken")
        contract_node = create_mock_node(
            cs.TS_SOL_CONTRACT_DECLARATION,
            fields={cs.FIELD_NAME: contract_name_node},
            children=[func_node]
        )
        func_node.node_parent = contract_node
        source_file = create_mock_node(
            cs.TS_SOL_SOURCE_FILE,
            children=[contract_node]
        )
        contract_node.node_parent = source_file

        file_path = Path("/repo/src/tokens/MyToken.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        result = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert result == "myproject.tokens.MyToken.MyToken.mint"


@pytest.mark.skipif(not SOLIDITY_PARSER_AVAILABLE, reason="tree-sitter-solidity not available")
class TestSolidityFQNWithRealParser:
    """Integration tests using real tree-sitter parsing for FQN generation.

    FQN Format: {project}.{module_parts}.{scope_name}.{function_name}

    The module_parts includes the file name (without extension) from the path.
    The scope_name is the contract/interface/library name containing the function.
    """

    def test_contract_fqn_real_parsing(self, solidity_parser: "Parser") -> None:
        """Test FQN for contract with real parser."""
        code = b"""
contract MyContract {
    function getValue() public view returns (uint) {
        return value;
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find contract node
        contract_node = None
        func_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                contract_node = child
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                            func_node = body_child
                            break
                break

        assert contract_node is not None
        assert func_node is not None

        file_path = Path("/repo/src/MyContract.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        # Function FQN: project.module.ContractName.functionName
        func_fqn = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert func_fqn == "myproject.MyContract.MyContract.getValue"

    def test_constructor_fqn_real_parsing(self, solidity_parser: "Parser") -> None:
        """Test FQN for constructor with real parser."""
        code = b"""
contract Owned {
    address public owner;

    constructor() {
        owner = msg.sender;
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find constructor node
        constructor_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_CONSTRUCTOR_DEFINITION:
                            constructor_node = body_child
                            break
                break

        assert constructor_node is not None

        file_path = Path("/repo/src/Owned.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            constructor_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert fqn == "myproject.Owned.Owned.constructor"

    def test_receive_fqn_real_parsing(self, solidity_parser: "Parser") -> None:
        """Test FQN for receive function with real parser."""
        code = b"""
contract Receiver {
    receive() external payable {
        // Handle ETH receive
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find receive node
        receive_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION:
                            receive_node = body_child
                            break
                break

        assert receive_node is not None

        file_path = Path("/repo/src/Receiver.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            receive_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert fqn == "myproject.Receiver.Receiver.receive"

    def test_fallback_fqn_real_parsing(self, solidity_parser: "Parser") -> None:
        """Test FQN for fallback function with real parser."""
        code = b"""
contract FallbackHandler {
    fallback() external {
        // Handle unknown calls
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find fallback node (no payable modifier)
        fallback_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION:
                            fallback_node = body_child
                            break
                break

        assert fallback_node is not None

        file_path = Path("/repo/src/FallbackHandler.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            fallback_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert fqn == "myproject.FallbackHandler.FallbackHandler.fallback"

    def test_library_function_fqn_real_parsing(self, solidity_parser: "Parser") -> None:
        """Test FQN for library function with real parser."""
        code = b"""
library SafeMath {
    function add(uint a, uint b) internal pure returns (uint) {
        return a + b;
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find function node inside library
        func_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_LIBRARY_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                            func_node = body_child
                            break
                break

        assert func_node is not None

        file_path = Path("/repo/src/libraries/SafeMath.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert fqn == "myproject.libraries.SafeMath.SafeMath.add"

    def test_interface_function_fqn_real_parsing(self, solidity_parser: "Parser") -> None:
        """Test FQN for interface function with real parser."""
        code = b"""
interface IERC20 {
    function transfer(address to, uint amount) external returns (bool);
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find function node inside interface
        func_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_INTERFACE_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                            func_node = body_child
                            break
                break

        assert func_node is not None

        file_path = Path("/repo/src/interfaces/IERC20.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert fqn == "myproject.interfaces.IERC20.IERC20.transfer"

    def test_modifier_fqn_real_parsing(self, solidity_parser: "Parser") -> None:
        """Test FQN for modifier with real parser."""
        code = b"""
contract Ownable {
    address public owner;

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find modifier node
        modifier_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_MODIFIER_DEFINITION:
                            modifier_node = body_child
                            break
                break

        assert modifier_node is not None

        file_path = Path("/repo/src/Ownable.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            modifier_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert fqn == "myproject.Ownable.Ownable.onlyOwner"

    def test_nested_path_fqn_real_parsing(self, solidity_parser: "Parser") -> None:
        """Test FQN with nested file path."""
        code = b"""
contract MyToken {
    function mint(address to, uint amount) public {
        // mint implementation
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find function node
        func_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                            func_node = body_child
                            break
                break

        assert func_node is not None

        # Use nested path
        file_path = Path("/repo/src/tokens/MyToken.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert fqn == "myproject.tokens.MyToken.MyToken.mint"

    def test_multiple_functions_same_contract(self, solidity_parser: "Parser") -> None:
        """Test FQN for multiple functions in same contract."""
        code = b"""
contract MultiFunction {
    function funcA() public pure returns (uint) {
        return 1;
    }

    function funcB() public pure returns (uint) {
        return 2;
    }

    function funcC() public pure returns (uint) {
        return 3;
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        file_path = Path("/repo/src/MultiFunction.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        # Find all functions and check their FQNs
        fqns = []
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                            fqn = resolve_fqn_from_ast(
                                body_child, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
                            )
                            if fqn:
                                fqns.append(fqn)

        assert "myproject.MultiFunction.MultiFunction.funcA" in fqns
        assert "myproject.MultiFunction.MultiFunction.funcB" in fqns
        assert "myproject.MultiFunction.MultiFunction.funcC" in fqns


@pytest.mark.skipif(not SOLIDITY_PARSER_AVAILABLE, reason="tree-sitter-solidity not available")
class TestSolidityFQNEdgeCases:
    """Tests for edge cases in Solidity FQN generation."""

    def test_abstract_contract_fqn(self, solidity_parser: "Parser") -> None:
        """Test FQN for abstract contract function."""
        code = b"""
abstract contract AbstractBase {
    function abstractMethod() public virtual returns (uint);
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find function node
        func_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                            func_node = body_child
                            break
                break

        assert func_node is not None

        file_path = Path("/repo/src/AbstractBase.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert fqn == "myproject.AbstractBase.AbstractBase.abstractMethod"

    def test_contract_with_inheritance_fqn(self, solidity_parser: "Parser") -> None:
        """Test FQN for contract with inheritance."""
        code = b"""
contract Child is Parent {
    function childMethod() public returns (uint) {
        return 42;
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find function node
        func_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                            func_node = body_child
                            break
                break

        assert func_node is not None

        file_path = Path("/repo/src/Child.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        # FQN uses Child (the contract containing the function), not Parent
        assert fqn == "myproject.Child.Child.childMethod"

    def test_contract_with_multiple_inheritance_fqn(self, solidity_parser: "Parser") -> None:
        """Test FQN for contract with multiple inheritance."""
        code = b"""
contract MultiInherit is A, B, C {
    function multiMethod() public returns (uint) {
        return 42;
    }
}
"""
        tree = solidity_parser.parse(code)
        root = tree.root_node

        # Find function node
        func_node = None
        for child in root.children:
            if child.type == cs.TS_SOL_CONTRACT_DECLARATION:
                body = child.child_by_field_name("body")
                if body:
                    for body_child in body.children:
                        if body_child.type == cs.TS_SOL_FUNCTION_DEFINITION:
                            func_node = body_child
                            break
                break

        assert func_node is not None

        file_path = Path("/repo/src/MultiInherit.sol")
        repo_root = Path("/repo")
        project_name = "myproject"

        fqn = resolve_fqn_from_ast(
            func_node, file_path, repo_root, project_name, SOLIDITY_FQN_SPEC
        )
        assert fqn == "myproject.MultiInherit.MultiInherit.multiMethod"