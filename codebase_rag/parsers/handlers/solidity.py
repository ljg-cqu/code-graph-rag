from __future__ import annotations

from typing import TYPE_CHECKING

from ... import constants as cs
from ..utils import safe_decode_text
from .base import BaseLanguageHandler

if TYPE_CHECKING:
    from ...types_defs import ASTNode


class SolidityHandler(BaseLanguageHandler):
    __slots__ = ()

    def is_class_method(self, node: ASTNode) -> bool:
        """Check if function is inside a contract (class method)."""
        return self.is_contract_function(node)

    def extract_function_name(self, node: ASTNode) -> str | None:
        """Extract function name, handling special cases."""
        if node.type == cs.TS_SOL_CONSTRUCTOR_DEFINITION:
            return "constructor"
        if node.type == cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION:
            # Determine if this is receive or fallback based on state mutability
            # receive() has payable modifier, fallback() does not
            if self.is_payable(node):
                return "receive"
            return "fallback"
        name_node = node.child_by_field_name(cs.FIELD_NAME)
        if name_node and name_node.text:
            return safe_decode_text(name_node)
        return None

    def extract_decorators(self, node: ASTNode) -> list[str]:
        """Extract modifiers used on functions."""
        modifiers = []
        for child in node.children:
            if child.type == cs.TS_SOL_MODIFIER_INVOCATION:
                # Extract the identifier child for the modifier name
                for subchild in child.children:
                    if subchild.type == cs.TS_SOL_IDENTIFIER:
                        if mod_text := safe_decode_text(subchild):
                            modifiers.append(mod_text)
                        break
        return modifiers

    def is_contract_function(self, node: ASTNode) -> bool:
        """Check if function is inside a contract."""
        current = node.parent
        while current:
            if current.type == cs.TS_SOL_CONTRACT_DECLARATION:
                return True
            if current.type == cs.TS_SOL_SOURCE_FILE:
                return False
            current = current.parent
        return False

    def extract_visibility(self, node: ASTNode) -> str | None:
        """Extract visibility modifier (public, private, internal, external).

        Note: In tree-sitter-solidity, visibility is a child node type, not a field.
        We iterate over children to find nodes with type 'visibility'.
        """
        for child in node.children:
            if child.type == cs.TS_SOL_VISIBILITY:
                return safe_decode_text(child)
        return None

    def extract_state_mutability(self, node: ASTNode) -> str | None:
        """Extract state mutability (view, pure, payable).

        Note: In tree-sitter-solidity, state_mutability is a child node type, not a field.
        We iterate over children to find nodes with type 'state_mutability'.
        """
        for child in node.children:
            if child.type == cs.TS_SOL_STATE_MUTABILITY:
                return safe_decode_text(child)
        return None

    def is_payable(self, node: ASTNode) -> bool:
        """Check if function is payable."""
        return self.extract_state_mutability(node) == "payable"

    def is_view_or_pure(self, node: ASTNode) -> bool:
        """Check if function is view or pure."""
        mutability = self.extract_state_mutability(node)
        return mutability in ("view", "pure")

    def extract_inheritance(self, node: ASTNode) -> list[str]:
        """Extract inherited contracts/interfaces from inheritance specifier."""
        bases = []
        for child in node.children:
            if child.type == cs.TS_SOL_INHERITANCE_SPECIFIER:
                for base in child.children:
                    if base.type == cs.TS_SOL_USER_DEFINED_TYPE:
                        # Extract identifier from user_defined_type node
                        for subchild in base.children:
                            if subchild.type == cs.TS_SOL_IDENTIFIER:
                                if base_text := safe_decode_text(subchild):
                                    bases.append(base_text)
                                break
        return bases

    def extract_base_class_name(self, base_node: ASTNode) -> str | None:
        """Extract base contract name from user_defined_type node."""
        if base_node.type == cs.TS_SOL_USER_DEFINED_TYPE:
            for child in base_node.children:
                if child.type == cs.TS_SOL_IDENTIFIER:
                    return safe_decode_text(child)
        return safe_decode_text(base_node) if base_node.text else None
