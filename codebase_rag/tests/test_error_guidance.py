"""Tests for the LLM-First error guidance system."""

from codebase_rag.services.error_guidance import (
    STATIC_GUIDANCE,
    ErrorContext,
    ErrorGuidance,
    LLMErrorGuidance,
    UserExpertiseLevel,
    format_user_error,
)
from codebase_rag.services.failure_classifier import FailureClassification, FailureType


class TestErrorContext:
    """Tests for ErrorContext dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """Should create with required fields."""
        context = ErrorContext(
            operation_type="document_indexing",
            error_category=FailureType.TRANSIENT_NETWORK,
        )

        assert context.operation_type == "document_indexing"
        assert context.error_category == FailureType.TRANSIENT_NETWORK
        assert context.graph_type == "code"
        assert context.user_expertise == UserExpertiseLevel.INTERMEDIATE

    def test_to_llm_prompt_sanitizes_data(self) -> None:
        """Should generate sanitized prompt context."""
        context = ErrorContext(
            operation_type="document_indexing",
            error_category=FailureType.TRANSIENT_NETWORK,
            embedding_provider="openai",
            previous_provider="local",
            recent_changes=["Changed EMBEDDING_PROVIDER from local to openai"],
        )

        prompt = context.to_llm_prompt()

        assert "Operation: document_indexing" in prompt
        assert "Error Type: TRANSIENT_NETWORK" in prompt
        assert "Embedding Provider: openai" in prompt
        assert "Previous Provider: local" in prompt


class TestErrorGuidance:
    """Tests for ErrorGuidance dataclass."""

    def test_creates_with_all_fields(self) -> None:
        """Should create with all fields."""
        guidance = ErrorGuidance(
            summary="Connection Failed",
            explanation="The database is not running.",
            suggested_fix="Start Memgraph with docker.",
            severity="blocking",
            code_example="docker run memgraph/memgraph",
            should_retry=True,
            retry_after_seconds=30,
        )

        assert guidance.summary == "Connection Failed"
        assert guidance.explanation == "The database is not running."
        assert guidance.suggested_fix == "Start Memgraph with docker."
        assert guidance.severity == "blocking"
        assert guidance.code_example == "docker run memgraph/memgraph"
        assert guidance.should_retry is True
        assert guidance.retry_after_seconds == 30


class TestStaticGuidance:
    """Tests for STATIC_GUIDANCE dictionary."""

    def test_has_import_target_missing_guidance(self) -> None:
        """Should have guidance for IMPORT_TARGET_MISSING."""
        assert "IMPORT_TARGET_MISSING" in STATIC_GUIDANCE
        guidance = STATIC_GUIDANCE["IMPORT_TARGET_MISSING"]
        assert guidance.severity == "info"
        assert guidance.should_retry is False
        assert "external" in guidance.explanation.lower()

    def test_has_mgclient_state_corruption_guidance(self) -> None:
        """Should have guidance for MGCLIENT_STATE_CORRUPTION."""
        assert "MGCLIENT_STATE_CORRUPTION" in STATIC_GUIDANCE
        guidance = STATIC_GUIDANCE["MGCLIENT_STATE_CORRUPTION"]
        assert guidance.severity == "warning"
        assert guidance.should_retry is True
        assert guidance.retry_after_seconds == 1

    def test_has_transient_network_guidance(self) -> None:
        """Should have guidance for TRANSIENT_NETWORK."""
        assert "TRANSIENT_NETWORK" in STATIC_GUIDANCE
        guidance = STATIC_GUIDANCE["TRANSIENT_NETWORK"]
        assert guidance.should_retry is True

    def test_has_vector_dimension_mismatch_guidance(self) -> None:
        """Should have guidance for VECTOR_DIMENSION_MISMATCH."""
        assert "VECTOR_DIMENSION_MISMATCH" in STATIC_GUIDANCE
        guidance = STATIC_GUIDANCE["VECTOR_DIMENSION_MISMATCH"]
        assert guidance.severity == "blocking"

    def test_has_enterprise_feature_required_guidance(self) -> None:
        """Should have guidance for ENTERPRISE_FEATURE_REQUIRED."""
        assert "ENTERPRISE_FEATURE_REQUIRED" in STATIC_GUIDANCE
        guidance = STATIC_GUIDANCE["ENTERPRISE_FEATURE_REQUIRED"]
        assert guidance.severity == "blocking"
        assert guidance.should_retry is False
        assert "enterprise" in guidance.explanation.lower()
        assert guidance.doc_link == "https://memgraph.com/enterprise"


class TestLLMErrorGuidance:
    """Tests for LLMErrorGuidance class."""

    def test_returns_static_guidance_without_llm(self) -> None:
        """Should return static guidance when no LLM is configured."""
        generator = LLMErrorGuidance(model_call=None)
        error = Exception("Connection reset by peer")
        context = ErrorContext(
            operation_type="document_indexing",
            error_category=FailureType.TRANSIENT_NETWORK,
        )
        classification = FailureClassification(
            failure_type=FailureType.TRANSIENT_NETWORK,
            message="Connection reset",
            should_retry=True,
            max_retries=3,
        )

        guidance = generator._get_static_guidance(error, context, classification)

        assert guidance.summary == "Connection Issue"
        assert guidance.should_retry is True

    def test_returns_dimension_mismatch_guidance(self) -> None:
        """Should return dimension mismatch guidance for embedding errors."""
        generator = LLMErrorGuidance(model_call=None)
        error = Exception("Dimension mismatch: expected 1536, got 768")
        context = ErrorContext(
            operation_type="embedding",
            error_category=FailureType.VECTOR_DIMENSION_MISMATCH,
            embedding_provider="openai",
            previous_provider="local",
        )
        classification = FailureClassification(
            failure_type=FailureType.VECTOR_DIMENSION_MISMATCH,
            message="Dimension mismatch",
            should_retry=False,
        )

        guidance = generator._get_static_guidance(error, context, classification)

        assert "Dimension" in guidance.summary
        assert "switched" in guidance.explanation.lower() or "mismatch" in guidance.explanation.lower()

    def test_returns_import_target_missing_guidance(self) -> None:
        """Should return import target missing guidance."""
        generator = LLMErrorGuidance(model_call=None)
        error = Exception("Import target not found")
        context = ErrorContext(
            operation_type="ingestion",
            error_category=FailureType.IMPORT_TARGET_MISSING,
        )
        classification = FailureClassification(
            failure_type=FailureType.IMPORT_TARGET_MISSING,
            message="Import target missing",
            should_retry=False,
        )

        guidance = generator._get_static_guidance(error, context, classification)

        assert "Import" in guidance.summary
        assert guidance.severity == "info"

    def test_returns_mgclient_state_corruption_guidance(self) -> None:
        """Should return mgclient state corruption guidance."""
        generator = LLMErrorGuidance(model_call=None)
        error = Exception("mgclient.Column returned a result with an exception set")
        context = ErrorContext(
            operation_type="query",
            error_category=FailureType.MGCLIENT_STATE_CORRUPTION,
        )
        classification = FailureClassification(
            failure_type=FailureType.MGCLIENT_STATE_CORRUPTION,
            message="Connection state corrupted",
            should_retry=True,
        )

        guidance = generator._get_static_guidance(error, context, classification)

        assert "Connection State" in guidance.summary
        assert guidance.should_retry is True


class TestFormatUserError:
    """Tests for format_user_error function."""

    def test_formats_basic_guidance(self) -> None:
        """Should format guidance for display."""
        guidance = ErrorGuidance(
            summary="Connection Failed",
            explanation="The database is not running.",
            suggested_fix="Start Memgraph with docker.",
        )
        error = Exception("Connection refused")

        formatted = format_user_error(guidance, error)

        assert "Connection Failed" in formatted
        assert "The database is not running" in formatted
        assert "Start Memgraph with docker" in formatted

    def test_includes_code_example(self) -> None:
        """Should include code example when present."""
        guidance = ErrorGuidance(
            summary="Connection Failed",
            explanation="Database not running.",
            suggested_fix="Start Memgraph.",
            code_example="docker run memgraph/memgraph",
        )
        error = Exception("Connection refused")

        formatted = format_user_error(guidance, error)

        assert "docker run memgraph/memgraph" in formatted

    def test_includes_technical_details_when_requested(self) -> None:
        """Should include technical details when show_technical is True."""
        guidance = ErrorGuidance(
            summary="Error",
            explanation="Something went wrong.",
            suggested_fix="Try again.",
        )
        error = Exception("Detailed technical error message")

        formatted = format_user_error(guidance, error, show_technical=True)

        assert "Technical details" in formatted
        assert "Detailed technical error message" in formatted

    def test_hides_technical_details_by_default(self) -> None:
        """Should hide technical details by default."""
        guidance = ErrorGuidance(
            summary="Error",
            explanation="Something went wrong.",
            suggested_fix="Try again.",
        )
        error = Exception("Detailed technical error message")

        formatted = format_user_error(guidance, error, show_technical=False)

        assert "Technical details" not in formatted
        assert "Detailed technical error message" not in formatted
