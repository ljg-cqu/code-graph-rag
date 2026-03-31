"""AutoHotkey language handler for Code-Graph-RAG.

This module provides language-specific handling for AutoHotkey scripts,
using the alfredomtx/tree-sitter-autohotkey grammar.

Handles:
- Hotkey/hotstring detection
- AHK v1 syntax (this grammar targets v1)
- GUI control tracking
- Command vs function call resolution
- Label-based flow control
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import constants as cs
from ..utils import safe_decode_text
from .base import BaseLanguageHandler

if TYPE_CHECKING:
    from ...types_defs import ASTNode


class AutoHotkeyHandler(BaseLanguageHandler):
    """Language handler for AutoHotkey scripts.

    Grammar: alfredomtx/tree-sitter-autohotkey (AHK v1)
    """

    __slots__ = ("_version",)

    def __init__(self) -> None:
        super().__init__()
        self._version: int | None = None

    def is_class_method(self, node: ASTNode) -> bool:
        """Check if function is inside a class."""
        current = node.parent
        while current:
            if current.type == cs.TS_AHK_CLASS_DEFINITION:
                return True
            if current.type == cs.TS_AHK_SOURCE_FILE:
                return False
            current = current.parent
        return False

    def extract_function_name(self, node: ASTNode) -> str | None:
        """Extract function name, handling hotkeys/hotstrings/labels."""
        if node.type == cs.TS_AHK_HOTKEY:
            # Hotkey is a token - get text directly (e.g., "^a::")
            text = safe_decode_text(node)
            if text:
                # Strip "::" to get the key combo
                return text.rstrip("::")
            return None

        if node.type == cs.TS_AHK_HOTSTRING_DEFINITION:
            trigger_node = node.child_by_field_name(cs.FIELD_TRIGGER)
            return safe_decode_text(trigger_node) if trigger_node else None

        if node.type == cs.TS_AHK_LABEL:
            # Label: identifier followed by colon
            for child in node.children:
                if child.type == cs.TS_AHK_IDENTIFIER:
                    return safe_decode_text(child)
            return None

        return super().extract_function_name(node)

    def is_hotkey(self, node: ASTNode) -> bool:
        """Check if node is a hotkey definition."""
        return node.type == cs.TS_AHK_HOTKEY

    def is_hotstring(self, node: ASTNode) -> bool:
        """Check if node is a hotstring definition."""
        return node.type == cs.TS_AHK_HOTSTRING_DEFINITION

    def is_label(self, node: ASTNode) -> bool:
        """Check if node is a label definition."""
        return node.type == cs.TS_AHK_LABEL

    def is_command(self, node: ASTNode) -> bool:
        """Check if node is an AHK command (vs function call)."""
        return node.type in (
            cs.TS_AHK_COMMAND,
            cs.TS_AHK_GUI_ACTION,
            cs.TS_AHK_GUI_ACTION_SPACED,
        )

    def extract_hotkey_modifiers(self, node: ASTNode) -> list[str]:
        """Extract modifier keys from hotkey definition.

        Common modifiers:
        - ^ = Ctrl
        - ! = Alt
        - + = Shift
        - # = Win
        - < = Left variant
        - > = Right variant

        Example: "^!s" = Ctrl+Alt+s
        """
        if node.type != cs.TS_AHK_HOTKEY:
            return []

        hotkey_text = safe_decode_text(node) or ""
        # Strip trailing "::"
        hotkey_text = hotkey_text.rstrip("::")

        modifier_map = {
            "^": "Ctrl",
            "!": "Alt",
            "+": "Shift",
            "#": "Win",
            "<": "Left",
            ">": "Right",
            "*": "Wildcard",
            "~": "PassThrough",
            "$": "Hook",
        }

        modifiers = []
        for char in hotkey_text:
            if char in modifier_map:
                modifiers.append(modifier_map[char])

        return modifiers

    def extract_hotkey_key(self, node: ASTNode) -> str | None:
        """Extract the base key from hotkey definition.

        Example: "^!s" -> "s"
                 "F1" -> "F1"
        """
        if node.type != cs.TS_AHK_HOTKEY:
            return None

        hotkey_text = safe_decode_text(node) or ""
        hotkey_text = hotkey_text.rstrip("::")

        # Remove modifiers to get base key
        base_key = hotkey_text.lstrip("^!+#<>*~$")
        return base_key if base_key else None

    def extract_hotstring_options(self, node: ASTNode) -> list[str]:
        """Extract hotstring options.

        Example: ":C*:btw::by the way" -> ["C", "*"]

        Common options:
        - C = Case-sensitive
        - O = Omit ending character
        - * = Fire immediately
        - ? = Allow alphanumeric
        - Z = Reset keyboard state
        - B0 = Wait for ending character
        - Kn = Send key history
        - Pn = Priority
        """
        if node.type != cs.TS_AHK_HOTSTRING_DEFINITION:
            return []

        options_node = node.child_by_field_name(cs.FIELD_OPTIONS)
        if options_node:
            options_text = safe_decode_text(options_node) or ""
            # Options are like "C*" - split into individual chars
            return list(options_text)

        return []

    def extract_hotstring_replacement(self, node: ASTNode) -> str | None:
        """Extract hotstring replacement text.

        Example: ":btw::by the way" -> "by the way"
        """
        if node.type != cs.TS_AHK_HOTSTRING_DEFINITION:
            return None

        replacement_node = node.child_by_field_name(cs.FIELD_REPLACEMENT)
        return safe_decode_text(replacement_node) if replacement_node else None

    def extract_gui_controls(self, node: ASTNode) -> list[str]:
        """Extract GUI control references.

        Example: Gui, Add, Button, Click me
        """
        controls = []
        if node.type in (cs.TS_AHK_GUI_ACTION, cs.TS_AHK_GUI_ACTION_SPACED):
            for child in node.children:
                if child.type == cs.TS_AHK_STRING:
                    control_type = safe_decode_text(child)
                    if control_type:
                        controls.append(control_type)
        return controls

    def detect_version(self, node: ASTNode) -> int:
        """Detect AHK version from script content.

        This grammar is for AHK v1, so default to v1.
        Check for #Requires directive if present.
        """
        if self._version is not None:
            return self._version

        # Check directives in source file
        current = node
        while current.parent:
            current = current.parent

        if current.type == cs.TS_AHK_SOURCE_FILE:
            for child in current.children:
                if child.type == cs.TS_AHK_DIRECTIVE:
                    text = safe_decode_text(child) or ""
                    if cs.AHK_V2_DIRECTIVE in text:
                        self._version = 2
                        return 2
                    if cs.AHK_V1_DIRECTIVE in text:
                        self._version = 1
                        return 1

        # Default to v1 (this grammar targets v1)
        self._version = 1
        return 1

    def is_ahk_v2(self, node: ASTNode) -> bool:
        """Check if script uses AHK v2 syntax."""
        return self.detect_version(node) == 2

    def extract_command_name(self, node: ASTNode) -> str | None:
        """Extract AHK command name.

        Commands are built-in AHK functions like:
        - Send, Click, MouseMove
        - Gui, Menu, Tray
        - FileRead, FileAppend
        - RegRead, RegWrite
        """
        if not self.is_command(node):
            return None

        name_node = node.child_by_field_name(cs.FIELD_NAME)
        if name_node and name_node.text:
            return safe_decode_text(name_node)

        # Fallback: first identifier
        for child in node.children:
            if child.type == cs.TS_AHK_IDENTIFIER:
                return safe_decode_text(child)

        return None

    def extract_class_name(self, node: ASTNode) -> str | None:
        """Extract class name from class_definition node.

        Args:
            node: AST node (should be class_definition)

        Returns:
            Class name or None if not a class node
        """
        if node.type != cs.TS_AHK_CLASS_DEFINITION:
            return None
        name_node = node.child_by_field_name(cs.FIELD_NAME)
        return safe_decode_text(name_node) if name_node else None