"""Tests for JSON ingestion file filtering.

This module tests the enhanced file filtering in _load_json_files_with_errors
to skip venv, node_modules, and non-CGR JSON files.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from codebase_rag.json_ingestion import _load_json_files_with_errors


class TestJsonFileFiltering:
    """Test JSON file filtering in _load_json_files_with_errors."""

    def test_filters_venv_files(self, tmp_path: Path):
        """Test that files in .venv are filtered out."""
        # Create a file in .venv
        venv_dir = tmp_path / ".venv" / "lib" / "site-packages"
        venv_dir.mkdir(parents=True)
        venv_file = venv_dir / "test.json"
        venv_file.write_text('{"entities": []}')

        json_files, _, skip_count = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 0

    def test_filters_site_packages_files(self, tmp_path: Path):
        """Test that files in site-packages are filtered out."""
        site_pkg_dir = tmp_path / "site-packages" / "botocore"
        site_pkg_dir.mkdir(parents=True)
        pkg_file = site_pkg_dir / "test.json"
        pkg_file.write_text('{"entities": []}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 0

    def test_filters_node_modules(self, tmp_path: Path):
        """Test that files in node_modules are filtered out."""
        nm_dir = tmp_path / "node_modules" / "lodash"
        nm_dir.mkdir(parents=True)
        nm_file = nm_dir / "package.json"
        nm_file.write_text('{"name": "lodash"}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 0

    def test_filters_aws_paginators(self, tmp_path: Path):
        """Test that AWS paginator files are filtered out."""
        aws_file = tmp_path / "paginators-1.json"
        aws_file.write_text('{"pagination": {}}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len([f for f, _ in json_files if "paginators" in str(f)]) == 0

    def test_filters_aws_examples(self, tmp_path: Path):
        """Test that AWS example files are filtered out."""
        aws_file = tmp_path / "examples-1.json"
        aws_file.write_text('{"examples": {}}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len([f for f, _ in json_files if "examples" in str(f)]) == 0

    def test_filters_aws_service_files(self, tmp_path: Path):
        """Test that AWS service files are filtered out."""
        aws_file = tmp_path / "service-2.json"
        aws_file.write_text('{"service": {}}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len([f for f, _ in json_files if "service" in str(f)]) == 0

    def test_allows_valid_cgr_files(self, tmp_path: Path):
        """Test that valid CGR JSON files are processed."""
        cgr_file = tmp_path / "entities.json"
        cgr_file.write_text('{"entities": [{"id": "1", "name": "test"}]}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 1
        assert json_files[0][0].name == "entities.json"

    def test_allows_cgr_in_dot_cgr_dir(self, tmp_path: Path):
        """Test that .cgr directory files are allowed."""
        cgr_dir = tmp_path / ".cgr"
        cgr_dir.mkdir()
        cgr_file = cgr_dir / "output.json"
        cgr_file.write_text('{"entities": []}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 1

    def test_filters_hidden_directories(self, tmp_path: Path):
        """Test that hidden directories (except .cgr) are filtered."""
        hidden_dir = tmp_path / ".hidden_dir"
        hidden_dir.mkdir()
        hidden_file = hidden_dir / "test.json"
        hidden_file.write_text('{"entities": []}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 0

    def test_allows_regular_directories(self, tmp_path: Path):
        """Test that regular directories are allowed."""
        regular_dir = tmp_path / "output"
        regular_dir.mkdir()
        regular_file = regular_dir / "entities.json"
        regular_file.write_text('{"entities": [{"id": "1", "name": "test"}]}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 1

    def test_filters_tmp_cache_files(self, tmp_path: Path):
        """Test that .tmp_cache_* files are filtered."""
        cache_file = tmp_path / ".tmp_cache_123.json"
        cache_file.write_text('{"entities": []}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 0

    def test_filters_cgr_cache_files(self, tmp_path: Path):
        """Test that .cgr-* cache files are filtered."""
        cache_file = tmp_path / ".cgr-hash-cache.json"
        cache_file.write_text('{"cache": {}}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 0

    def test_filters_embedding_cache_dir(self, tmp_path: Path):
        """Test that .embedding_cache dir is filtered."""
        cache_dir = tmp_path / ".embedding_cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "data.json"
        cache_file.write_text('{"vectors": []}')

        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 0

    def test_custom_exclude_patterns(self, tmp_path: Path):
        """Test that custom exclude patterns work."""
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        test_file = test_dir / "test.json"
        test_file.write_text('{"entities": []}')

        json_files, _, _ = _load_json_files_with_errors(
            str(tmp_path),
            exclude_patterns=["test_data/*", "test_data/**/*"]
        )
        assert len(json_files) == 0

    def test_combined_default_and_custom_patterns(self, tmp_path: Path):
        """Test that default and custom patterns both apply."""
        # Create file in venv (default pattern)
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        venv_file = venv_dir / "test.json"
        venv_file.write_text('{"entities": []}')

        # Create file in custom location
        custom_dir = tmp_path / "custom_skip"
        custom_dir.mkdir()
        custom_file = custom_dir / "test.json"
        custom_file.write_text('{"entities": []}')

        json_files, _, _ = _load_json_files_with_errors(
            str(tmp_path),
            exclude_patterns=["custom_skip/*", "custom_skip/**/*"]
        )
        assert len(json_files) == 0

    def test_preserves_existing_functionality(self, tmp_path: Path):
        """Test that existing functionality still works."""
        # Create valid CGR file
        valid_file = tmp_path / "valid.json"
        valid_file.write_text(json.dumps({
            "metadata": {"dataset_id": "test"},
            "entities": [{"id": "1", "name": "Entity1"}]
        }))

        # Create invalid JSON file (schema doesn't match)
        invalid_file = tmp_path / "invalid.txt"  # Not a JSON file
        invalid_file.write_text("not json")

        json_files, errors, skip_count = _load_json_files_with_errors(str(tmp_path))

        # Should find the valid file
        assert len(json_files) == 1
        assert json_files[0][0].name == "valid.json"


class TestFilterPresets:
    """Test filter preset functionality."""

    def test_lenient_preset_uses_default_filters(self, tmp_path: Path):
        """Test lenient preset uses default filters."""
        # Create file in venv (should be filtered)
        venv_dir = tmp_path / ".venv" / "lib"
        venv_dir.mkdir(parents=True)
        venv_file = venv_dir / "test.json"
        venv_file.write_text('{"entities": []}')

        # Create valid file in regular location
        valid_file = tmp_path / "valid.json"
        valid_file.write_text('{"entities": [{"id": "1", "name": "test"}]}')

        json_files, _, _ = _load_json_files_with_errors(
            str(tmp_path), filter_preset="lenient"
        )
        assert len(json_files) == 1
        assert json_files[0][0].name == "valid.json"

    def test_strict_preset_only_allows_cgr_json_files(self, tmp_path: Path):
        """Test strict preset only processes .cgr.json files."""
        # Create regular JSON file (should be filtered in strict mode)
        regular_file = tmp_path / "data.json"
        regular_file.write_text('{"entities": [{"id": "1", "name": "test"}]}')

        # Create .cgr.json file (should be processed)
        cgr_file = tmp_path / "data.cgr.json"
        cgr_file.write_text('{"entities": [{"id": "2", "name": "cgr_test"}]}')

        json_files, _, _ = _load_json_files_with_errors(
            str(tmp_path), filter_preset="strict"
        )
        assert len(json_files) == 1
        assert json_files[0][0].name == "data.cgr.json"

    def test_none_preset_disables_default_filters(self, tmp_path: Path):
        """Test none preset disables default filtering."""
        # Create file that would normally be filtered (e.g., in build dir)
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        build_file = build_dir / "output.json"
        build_file.write_text('{"entities": [{"id": "1", "name": "test"}]}')

        # With "none" preset, should process files in build dir
        json_files, _, _ = _load_json_files_with_errors(
            str(tmp_path), filter_preset="none"
        )
        # Note: hidden directories are still filtered by should_exclude()
        # but build/ is not hidden, so it should be included
        assert len(json_files) == 1
        assert json_files[0][0].name == "output.json"

    def test_default_preset_is_lenient(self, tmp_path: Path):
        """Test default preset is lenient."""
        # Create file in venv (should be filtered)
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        venv_file = venv_dir / "test.json"
        venv_file.write_text('{"entities": []}')

        # Create valid file
        valid_file = tmp_path / "valid.json"
        valid_file.write_text('{"entities": [{"id": "1", "name": "test"}]}')

        # Without specifying filter_preset, should use lenient
        json_files, _, _ = _load_json_files_with_errors(str(tmp_path))
        assert len(json_files) == 1
        assert json_files[0][0].name == "valid.json"
