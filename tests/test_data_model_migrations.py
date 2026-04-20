"""Unit tests for data model migration functions."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.migrations.data_model_migrations import (
    _migrate_external_module_paths,
    _migrate_json_entity_labels,
    _migrate_json_node_names,
    _migrate_method_is_exported,
    _migrate_orphaned_builtins,
    run_migrations,
)


class TestMigrateOrphanedBuiltins:
    """Test orphaned builtin migration."""

    def test_no_orphaned_builtins(self):
        """Test when no orphaned builtins exist."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)

        result = _migrate_orphaned_builtins(mock_cursor, dry_run=True)
        assert result == 0

    def test_dry_run_reports_count(self):
        """Test dry run reports count without making changes."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (48,)

        with patch("codebase_rag.migrations.data_model_migrations.logger") as mock_logger:
            result = _migrate_orphaned_builtins(mock_cursor, dry_run=True)

        assert result == 48
        assert mock_cursor.execute.call_count == 1  # Only count query
        mock_logger.info.assert_called_once()

    def test_execute_migrates_builtins(self):
        """Test actual migration creates builtin module and relationships."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (48,)

        with patch("codebase_rag.migrations.data_model_migrations.logger"):
            result = _migrate_orphaned_builtins(mock_cursor, dry_run=False)

        assert result == 48
        assert mock_cursor.execute.call_count == 3  # Count, MERGE module, MERGE relationship


class TestMigrateExternalModulePaths:
    """Test external module path migration."""

    def test_no_external_modules_with_path(self):
        """Test when no external modules have path set."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)

        result = _migrate_external_module_paths(mock_cursor, dry_run=True)
        assert result == 0

    def test_dry_run_reports_count(self):
        """Test dry run reports count without making changes."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (107,)

        with patch("codebase_rag.migrations.data_model_migrations.logger") as mock_logger:
            result = _migrate_external_module_paths(mock_cursor, dry_run=True)

        assert result == 107
        assert mock_cursor.execute.call_count == 1
        mock_logger.info.assert_called_once()

    def test_execute_migrates_paths(self):
        """Test actual migration updates paths to import_path."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (107,)

        with patch("codebase_rag.migrations.data_model_migrations.logger"):
            result = _migrate_external_module_paths(mock_cursor, dry_run=False)

        assert result == 107
        assert mock_cursor.execute.call_count == 2  # Count and UPDATE


class TestMigrateJsonNodeNames:
    """Test JSON node name migration."""

    def test_no_json_nodes_missing_name(self):
        """Test when all JSON nodes have names."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(0,), (0,), (0,), (0,)]  # All counts 0

        result = _migrate_json_node_names(mock_cursor, dry_run=True)
        assert result == 0

    def test_dry_run_reports_total_count(self):
        """Test dry run reports total count across all JSON types."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            (515,),   # JsonObject
            (170,),   # JsonArray
            (2592,),  # JsonField
            (2482,),  # JsonValue
        ]

        with patch("codebase_rag.migrations.data_model_migrations.logger") as mock_logger:
            result = _migrate_json_node_names(mock_cursor, dry_run=True)

        assert result == 5759
        assert mock_cursor.execute.call_count == 4  # Four count queries
        mock_logger.info.assert_called_once()

    def test_execute_sets_names(self):
        """Test actual migration sets names for all JSON types."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            (10,),   # JsonObject
            (5,),    # JsonArray
            (20,),   # JsonField
            (15,),   # JsonValue
        ]

        with patch("codebase_rag.migrations.data_model_migrations.logger"):
            result = _migrate_json_node_names(mock_cursor, dry_run=False)

        assert result == 50
        assert mock_cursor.execute.call_count == 8  # 4 counts + 4 updates


class TestMigrateMethodIsExported:
    """Test method is_exported migration."""

    def test_no_methods_missing_is_exported(self):
        """Test when all methods have is_exported."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)

        result = _migrate_method_is_exported(mock_cursor, dry_run=True)
        assert result == 0

    def test_dry_run_reports_count(self):
        """Test dry run reports count without making changes."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (25,)

        with patch("codebase_rag.migrations.data_model_migrations.logger") as mock_logger:
            result = _migrate_method_is_exported(mock_cursor, dry_run=True)

        assert result == 25
        assert mock_cursor.execute.call_count == 1
        mock_logger.info.assert_called_once()

    def test_execute_sets_is_exported(self):
        """Test actual migration sets is_exported to false."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (25,)

        with patch("codebase_rag.migrations.data_model_migrations.logger"):
            result = _migrate_method_is_exported(mock_cursor, dry_run=False)

        assert result == 25
        assert mock_cursor.execute.call_count == 2  # Count and SET


class TestMigrateJsonEntityLabels:
    """Test JsonEntity labels property rename migration."""

    def test_no_json_entities_with_labels(self):
        """Test when no JsonEntity nodes have labels property."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)

        result = _migrate_json_entity_labels(mock_cursor, dry_run=True)
        assert result == 0

    def test_dry_run_reports_count(self):
        """Test dry run reports count without making changes."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100,)

        with patch("codebase_rag.migrations.data_model_migrations.logger") as mock_logger:
            result = _migrate_json_entity_labels(mock_cursor, dry_run=True)

        assert result == 100
        assert mock_cursor.execute.call_count == 1
        mock_logger.info.assert_called_once()

    def test_execute_renames_labels(self):
        """Test actual migration renames labels to entity_labels."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100,)

        with patch("codebase_rag.migrations.data_model_migrations.logger"):
            result = _migrate_json_entity_labels(mock_cursor, dry_run=False)

        assert result == 100
        assert mock_cursor.execute.call_count == 2  # Count and SET/REMOVE


class TestRunMigrations:
    """Test main run_migrations function."""

    def test_dry_run_returns_all_counts(self):
        """Test dry run returns counts for all migrations."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # Order matches run_migrations: orphaned_builtins, external_module_paths,
        # json_node_names (4 counts), method_is_exported, json_entity_labels
        mock_cursor.fetchone.side_effect = [
            (48,),    # orphaned_builtins
            (107,),   # external_module_paths
            (10,), (5,), (20,), (15,),  # json_node_names (4 counts for 4 types)
            (25,),    # method_is_exported
            (100,),   # json_entity_labels
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("codebase_rag.migrations.data_model_migrations.mgclient.connect", return_value=mock_conn):
            with patch("codebase_rag.migrations.data_model_migrations.settings"):
                with patch("codebase_rag.migrations.data_model_migrations.logger"):
                    result = run_migrations(dry_run=True)

        assert result["orphaned_builtins"] == 48
        assert result["external_module_paths"] == 107
        assert result["json_node_names"] == 50  # 10 + 5 + 20 + 15
        assert result["method_is_exported"] == 25
        assert result["json_entity_labels"] == 100

    def test_connection_cleanup_on_success(self):
        """Test cursor and connection are closed after success."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # 8 fetchone calls: orphaned_builtins(1), external_module_paths(1),
        # json_node_names(4), method_is_exported(1), json_entity_labels(1)
        mock_cursor.fetchone.side_effect = [
            (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,),
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("codebase_rag.migrations.data_model_migrations.mgclient.connect", return_value=mock_conn):
            with patch("codebase_rag.migrations.data_model_migrations.settings"):
                with patch("codebase_rag.migrations.data_model_migrations.logger"):
                    run_migrations(dry_run=True)

        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_connection_cleanup_on_error(self):
        """Test cursor and connection are closed even on error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("DB error")
        mock_conn.cursor.return_value = mock_cursor

        with patch("codebase_rag.migrations.data_model_migrations.mgclient.connect", return_value=mock_conn):
            with patch("codebase_rag.migrations.data_model_migrations.settings"):
                with pytest.raises(Exception, match="DB error"):
                    run_migrations(dry_run=True)

        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()
