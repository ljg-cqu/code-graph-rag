"""Tests for vector index dimension type safety.

This module tests the enhanced _read_vector_index_dimension function
with improved type handling for float dimensions and validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.document.document_updater import (
    _read_vector_index_dimension,
    _validate_dimension,
)


class TestValidateDimension:
    """Test dimension validation helper."""

    def test_valid_dimension(self):
        """Test valid positive dimension passes."""
        result = _validate_dimension(768, "test_idx")
        assert result == 768

    def test_zero_dimension_rejected(self):
        """Test zero dimension is rejected."""
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _validate_dimension(0, "test_idx")
        assert result is None
        mock_logger.warning.assert_called_once()
        assert "must be positive" in mock_logger.warning.call_args[0][0]

    def test_negative_dimension_rejected(self):
        """Test negative dimension is rejected."""
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _validate_dimension(-1, "test_idx")
        assert result is None
        mock_logger.warning.assert_called_once()
        assert "must be positive" in mock_logger.warning.call_args[0][0]

    def test_dimension_too_large_rejected(self):
        """Test dimension exceeding max is rejected."""
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _validate_dimension(15000, "test_idx")
        assert result is None
        mock_logger.warning.assert_called_once()
        assert "exceeds max" in mock_logger.warning.call_args[0][0]

    def test_dimension_at_boundary_accepted(self):
        """Test dimension at max boundary is accepted."""
        result = _validate_dimension(10000, "test_idx")
        assert result == 10000


class TestReadVectorIndexDimension:
    """Test enhanced _read_vector_index_dimension function."""

    def test_valid_int_dimension(self):
        """Test int dimension passes through."""
        row = {"dimension": 768}
        result = _read_vector_index_dimension(row, "test_idx")
        assert result == 768

    def test_valid_string_dimension(self):
        """Test string dimension is parsed."""
        row = {"dimension": "768"}
        result = _read_vector_index_dimension(row, "test_idx")
        assert result == 768

    def test_valid_float_dimension(self):
        """Test whole-number float is converted."""
        row = {"dimension": 768.0}
        result = _read_vector_index_dimension(row, "test_idx")
        assert result == 768

    def test_none_row(self):
        """Test None row returns None."""
        result = _read_vector_index_dimension(None, "test_idx")
        assert result is None

    def test_missing_dimension(self):
        """Test missing dimension field returns None."""
        row = {"other_field": "value"}
        result = _read_vector_index_dimension(row, "test_idx")
        assert result is None

    def test_invalid_string_logs_warning(self):
        """Test invalid string logs warning."""
        row = {"dimension": "invalid"}
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _read_vector_index_dimension(row, "test_idx")
        assert result is None
        mock_logger.warning.assert_called_once()
        assert "invalid string dimension" in mock_logger.warning.call_args[0][0]

    def test_non_integer_float_rejected(self):
        """Test non-integer float is rejected with warning."""
        row = {"dimension": 768.5}
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _read_vector_index_dimension(row, "test_idx")
        assert result is None
        mock_logger.warning.assert_called_once()
        assert "non-integer float" in mock_logger.warning.call_args[0][0]

    def test_zero_dimension_rejected(self):
        """Test zero dimension is rejected."""
        row = {"dimension": 0}
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _read_vector_index_dimension(row, "test_idx")
        assert result is None
        # Should call warning for invalid dimension
        mock_logger.warning.assert_called()
        assert "must be positive" in mock_logger.warning.call_args[0][0]

    def test_negative_dimension_rejected(self):
        """Test negative dimension is rejected."""
        row = {"dimension": -1}
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _read_vector_index_dimension(row, "test_idx")
        assert result is None
        mock_logger.warning.assert_called()
        assert "must be positive" in mock_logger.warning.call_args[0][0]

    def test_large_dimension_rejected(self):
        """Test large dimension is rejected."""
        row = {"dimension": 50000}
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _read_vector_index_dimension(row, "test_idx")
        assert result is None
        mock_logger.warning.assert_called()
        assert "exceeds max" in mock_logger.warning.call_args[0][0]

    def test_unknown_type_logs_warning(self):
        """Test unknown type logs warning."""
        row = {"dimension": [1, 2, 3]}
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _read_vector_index_dimension(row, "test_idx")
        assert result is None
        mock_logger.warning.assert_called_once()
        log_msg = mock_logger.warning.call_args[0][0]
        assert "unexpected dimension type" in log_msg
        assert "list" in log_msg

    def test_index_name_in_logs(self):
        """Test index name appears in log messages."""
        row = {"dimension": "invalid"}
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _read_vector_index_dimension(row, "my_index")
        assert result is None
        mock_logger.warning.assert_called_once()
        assert "my_index" in mock_logger.warning.call_args[0][0]

    def test_float_conversion_logged(self):
        """Test float conversion is logged at debug level."""
        row = {"dimension": 512.0}
        with patch("codebase_rag.document.document_updater.logger") as mock_logger:
            result = _read_vector_index_dimension(row, "test_idx")
        assert result == 512
        mock_logger.debug.assert_called()
        assert "converted float" in mock_logger.debug.call_args[0][0]
