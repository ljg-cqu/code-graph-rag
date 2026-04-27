"""Unit tests for JSON graph quality migrations."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.migrations.json_graph_migrations import (
    LABELS_TO_NORMALIZE,
    RELATIONSHIP_TYPE_MAPPING,
    VALID_CATEGORIES,
    _escape_cypher_identifier,
    _migrate_entity_labels,
    _migrate_relationship_properties,
    _migrate_relationship_types,
    run_json_graph_migrations,
)


class TestEscapeCypherIdentifier:
    """Test Cypher identifier escaping."""

    def test_simple_identifier(self):
        """Test escaping simple identifier."""
        result = _escape_cypher_identifier("HIERARCHICAL")
        assert result == "`HIERARCHICAL`"

    def test_identifier_with_emoji(self):
        """Test escaping identifier with emoji."""
        result = _escape_cypher_identifier("🌳 is a")
        assert result == "`🌳 is a`"

    def test_identifier_with_backtick(self):
        """Test escaping identifier containing backtick."""
        result = _escape_cypher_identifier("test`value")
        assert result == "`test``value`"


class TestRelationshipTypeMapping:
    """Test relationship type mapping completeness."""

    def test_mapping_has_all_categories(self):
        """Test that mapping covers all canonical categories."""
        mapped_categories = {new_label for _, (new_label, _, _) in RELATIONSHIP_TYPE_MAPPING.items()}
        expected = {"HIERARCHICAL", "COMPOSITIONAL", "CAUSAL", "CONTEXTUAL", "ATTRIBUTIVE", "COMPARATIVE", "SEQUENTIAL", "ANALOGICAL"}
        assert expected.issubset(mapped_categories)

    def test_mapping_preserves_emoji(self):
        """Test that emoji is preserved in mapping."""
        for old_type, (new_label, verb, emoji) in RELATIONSHIP_TYPE_MAPPING.items():
            # Emoji should be at least one character
            assert len(emoji) >= 1
            # Old type should start with that emoji (use startswith, not [0], for multi-codepoint emojis)
            assert old_type.startswith(emoji), f"old_type={old_type!r} does not start with emoji={emoji!r}"

    def test_valid_categories_constant(self):
        """Test VALID_CATEGORIES includes all expected values."""
        expected = {"HIERARCHICAL", "COMPOSITIONAL", "CAUSAL", "CONTEXTUAL", "ATTRIBUTIVE", "COMPARATIVE", "SEQUENTIAL", "ANALOGICAL", "RELATED_TO"}
        assert VALID_CATEGORIES == expected


class TestMigrateRelationshipTypes:
    """Test relationship type migration."""

    def test_no_relationships_to_migrate(self):
        """Test when no relationships need migration."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)

        result = _migrate_relationship_types(mock_cursor, dry_run=True)
        assert result == 0

    def test_dry_run_reports_count(self):
        """Test dry run reports count without making changes.

        The function iterates over all relationship types. Use side_effect
        to return (10,) only for the first type and (0,) for the rest.
        """
        # 53 relationship types in RELATIONSHIP_TYPE_MAPPING
        # Only first one returns count > 0, rest return 0
        side_effect_values = [(10,)] + [(0,)] * 52
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = side_effect_values

        with patch("codebase_rag.migrations.json_graph_migrations.logger"):
            result = _migrate_relationship_types(mock_cursor, dry_run=True)

        assert result == 10
        # Each type: count query only in dry_run mode
        assert mock_cursor.execute.call_count == 53

    def test_execute_migrates_relationships(self):
        """Test actual migration creates new relationships.

        The function iterates over all relationship types. Use side_effect
        to return (2,) only for the first type and (0,) for the rest.
        """
        # 53 relationship types in RELATIONSHIP_TYPE_MAPPING
        # Only first one returns count > 0, rest return 0
        side_effect_values = [(2,)] + [(0,)] * 52
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = side_effect_values
        mock_cursor.fetchall.return_value = [
            ("source1", "target1", {"confidence": 0.9}),
            ("source2", "target2", {"confidence": 0.8}),
        ]

        with patch("codebase_rag.migrations.json_graph_migrations.logger"):
            result = _migrate_relationship_types(mock_cursor, dry_run=False)

        assert result == 2
        # 53 count queries (one per type) + 1 fetch all + 2 create + 2 delete = 58
        assert mock_cursor.execute.call_count == 58


class TestMigrateEntityLabels:
    """Test entity label normalization."""

    def test_no_labels_to_normalize(self):
        """Test when no labels need normalization."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)

        result = _migrate_entity_labels(mock_cursor, dry_run=True)
        assert result == 0

    def test_dry_run_reports_count(self):
        """Test dry run reports count without making changes.

        The function iterates over all labels to normalize. Use side_effect
        to return (7,) only for the first label and (0,) for the rest.
        """
        # 25 labels in LABELS_TO_NORMALIZE
        # Only first one returns count > 0, rest return 0
        side_effect_values = [(7,)] + [(0,)] * 24
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = side_effect_values

        with patch("codebase_rag.migrations.json_graph_migrations.logger"):
            result = _migrate_entity_labels(mock_cursor, dry_run=True)

        assert result == 7
        assert mock_cursor.execute.call_count == 25

    def test_execute_removes_labels(self):
        """Test actual migration removes space-containing labels.

        The function iterates over all labels. Use side_effect to return
        (7,) only for the first label and (0,) for the rest.
        """
        # 25 labels in LABELS_TO_NORMALIZE
        # Only first one returns count > 0, rest return 0
        side_effect_values = [(7,)] + [(0,)] * 24
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = side_effect_values

        with patch("codebase_rag.migrations.json_graph_migrations.logger"):
            result = _migrate_entity_labels(mock_cursor, dry_run=False)

        assert result == 7
        # 25 count queries + 1 REMOVE query = 26
        assert mock_cursor.execute.call_count == 26


class TestMigrateRelationshipProperties:
    """Test relationship property consolidation."""

    def test_no_properties_to_consolidate(self):
        """Test when no properties need consolidation."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)

        result = _migrate_relationship_properties(mock_cursor, dry_run=True)
        assert result == 0

    def test_dry_run_reports_count(self):
        """Test dry run reports count without making changes."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (185,)

        with patch("codebase_rag.migrations.json_graph_migrations.logger"):
            result = _migrate_relationship_properties(mock_cursor, dry_run=True)

        assert result == 185
        assert mock_cursor.execute.call_count == 1

    def test_execute_consolidates_properties(self):
        """Test actual migration consolidates properties."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (2,)
        mock_cursor.fetchall.return_value = [
            (1, {"category": "CAUSAL", "relationship_category": "CAUSAL", "relationship_emoji": "⚡"}),
            (2, {"category": "HIERARCHICAL", "category_with_emoji": "🌳 Hierarchical"}),
        ]

        with patch("codebase_rag.migrations.json_graph_migrations.logger"):
            result = _migrate_relationship_properties(mock_cursor, dry_run=False)

        assert result == 2


class TestRunJsonGraphMigrations:
    """Test main run_json_graph_migrations function."""

    def test_dry_run_returns_all_counts(self):
        """Test dry run returns counts for all migrations.

        The function calls:
        - _migrate_relationship_types: 53 fetchone calls (one per relationship type)
        - _migrate_entity_labels: 25 fetchone calls (one per label)
        - _migrate_relationship_properties: 1 fetchone call
        Total: 79 fetchone calls

        We mock first call returning (10,), second returning (7,), third returning (185,),
        then (0,) for remaining 76 calls.
        """
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # First call returns (10,) for relationship_types, second returns (7,) for entity_labels,
        # third returns (185,) for relationship_properties, rest return (0,)
        mock_cursor.fetchone.side_effect = [(10,), (7,), (185,)] + [(0,)] * 76
        mock_conn.cursor.return_value = mock_cursor

        with patch("codebase_rag.migrations.json_graph_migrations.mgclient.connect", return_value=mock_conn):
            with patch("codebase_rag.migrations.json_graph_migrations.settings"):
                with patch("codebase_rag.migrations.json_graph_migrations.logger"):
                    result = run_json_graph_migrations(dry_run=True)

        assert "relationship_types" in result
        assert "entity_labels" in result
        assert "relationship_properties" in result

    def test_connection_cleanup_on_success(self):
        """Test cursor and connection are closed after success."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # All fetchone calls return (0,) - no migrations needed
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.cursor.return_value = mock_cursor

        with patch("codebase_rag.migrations.json_graph_migrations.mgclient.connect", return_value=mock_conn):
            with patch("codebase_rag.migrations.json_graph_migrations.settings"):
                with patch("codebase_rag.migrations.json_graph_migrations.logger"):
                    run_json_graph_migrations(dry_run=True)

        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()


class TestLabelsToNormalize:
    """Test label normalization mapping."""

    def test_all_labels_have_spaces(self):
        """Test that all labels to normalize contain spaces."""
        for old_label in LABELS_TO_NORMALIZE.keys():
            assert " " in old_label, f"Label '{old_label}' should contain a space"

    def test_normalized_labels_are_pascal_case(self):
        """Test that normalized labels follow PascalCase."""
        for old_label, new_label in LABELS_TO_NORMALIZE.items():
            # PascalCase: no spaces, first char uppercase
            assert " " not in new_label
            assert new_label[0].isupper()
            # Normalized should be old_label without spaces
            assert new_label == old_label.replace(" ", "")

    def test_mapping_covers_verified_types(self):
        """Test that mapping covers types verified in graph."""
        verified_types = {
            "Dynamic Construct", "Governance Construct", "Abstract Class",
            "External Metaphor", "Emergent State", "Safety Boundary",
            "Progression Stage", "Foundational Construct", "Framework Component",
            "Regulatory Framework", "Role Archetype", "Planning Tool",
        }
        assert verified_types.issubset(LABELS_TO_NORMALIZE.keys())


class TestEdgeCases:
    """Test edge cases in migrations."""

    def test_empty_properties_in_relationship_migration(self):
        """Test handling of relationships with no properties.

        The function iterates over all relationship types. Use side_effect
        to return (1,) only for the first type and (0,) for the rest.
        """
        # 53 relationship types in RELATIONSHIP_TYPE_MAPPING
        # Only first one returns count > 0, rest return 0
        side_effect_values = [(1,)] + [(0,)] * 52
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = side_effect_values
        mock_cursor.fetchall.return_value = [
            ("source1", "target1", None),  # No properties
        ]

        with patch("codebase_rag.migrations.json_graph_migrations.logger"):
            result = _migrate_relationship_types(mock_cursor, dry_run=False)

        assert result == 1

    def test_missing_category_in_property_consolidation(self):
        """Test handling of relationships without category."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.fetchall.return_value = [
            (1, {"relationship_emoji": "⚡"}),  # No category
        ]

        with patch("codebase_rag.migrations.json_graph_migrations.logger"):
            result = _migrate_relationship_properties(mock_cursor, dry_run=False)

        assert result == 1

    def test_invalid_category_falls_back_to_related_to(self):
        """Test that invalid category falls back to RELATED_TO."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.fetchall.return_value = [
            (1, {"category": "INVALID_CATEGORY", "relationship_emoji": "❓"}),
        ]

        with patch("codebase_rag.migrations.json_graph_migrations.logger"):
            result = _migrate_relationship_properties(mock_cursor, dry_run=False)

        assert result == 1

    def test_connection_cleanup_on_error(self):
        """Test cursor and connection are closed even on error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("DB error")
        mock_conn.cursor.return_value = mock_cursor

        with patch("codebase_rag.migrations.json_graph_migrations.mgclient.connect", return_value=mock_conn):
            with patch("codebase_rag.migrations.json_graph_migrations.settings"):
                with patch("codebase_rag.migrations.json_graph_migrations.logger"):
                    with pytest.raises(Exception, match="DB error"):
                        run_json_graph_migrations(dry_run=True)

        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()
