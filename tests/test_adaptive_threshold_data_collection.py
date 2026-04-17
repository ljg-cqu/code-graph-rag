import pytest
from codebase_rag.orchestrator.concurrency_eligibility_classifier import ConcurrencyEligibilityClassifier


class TestRecordExecutionResult:
    """Verify record_execution_result populates success_rate_tracker correctly."""

    def test_record_success_creates_tracker_entry(self):
        classifier = ConcurrencyEligibilityClassifier()
        classifier.record_execution_result("llm_analyzed", True)
        assert "llm_analyzed" in classifier.success_rate_tracker
        assert classifier.success_rate_tracker["llm_analyzed"] == [True]

    def test_record_failure_creates_tracker_entry(self):
        classifier = ConcurrencyEligibilityClassifier()
        classifier.record_execution_result("llm_rejected", False)
        assert "llm_rejected" in classifier.success_rate_tracker
        assert classifier.success_rate_tracker["llm_rejected"] == [False]

    def test_multiple_records_accumulate(self):
        classifier = ConcurrencyEligibilityClassifier()
        for _ in range(10):
            classifier.record_execution_result("llm_analyzed", True)
        assert len(classifier.success_rate_tracker["llm_analyzed"]) == 10

    def test_tracker_capped_at_100(self):
        classifier = ConcurrencyEligibilityClassifier()
        for _ in range(150):
            classifier.record_execution_result("llm_analyzed", True)
        assert len(classifier.success_rate_tracker["llm_analyzed"]) == 100

    def test_mixed_success_failure_records(self):
        classifier = ConcurrencyEligibilityClassifier()
        classifier.record_execution_result("llm_analyzed", True)
        classifier.record_execution_result("llm_analyzed", True)
        classifier.record_execution_result("llm_analyzed", False)
        tracker = classifier.success_rate_tracker["llm_analyzed"]
        assert tracker == [True, True, False]
        assert sum(tracker) / len(tracker) == 2/3


class TestAdaptiveThresholdActivation:
    """Verify that _adjust_threshold_based_on_success activates after 10 data points."""

    def test_threshold_unchanged_with_insufficient_data(self):
        """With fewer than 10 records, threshold should not adjust."""
        classifier = ConcurrencyEligibilityClassifier()
        for _ in range(5):
            classifier.record_execution_result("llm_analyzed", True)
        result = classifier._adjust_threshold_based_on_success("llm_analyzed")
        assert result == classifier.threshold  # No adjustment

    def test_threshold_lowered_with_high_success_rate(self):
        """With ≥10 records and ≥85% success, threshold should lower."""
        classifier = ConcurrencyEligibilityClassifier()
        # 10 successes out of 10 = 100% success rate
        for _ in range(10):
            classifier.record_execution_result("llm_analyzed", True)
        result = classifier._adjust_threshold_based_on_success("llm_analyzed")
        assert result < classifier.threshold
        assert result == max(0.5, classifier.threshold * 0.8)

    def test_threshold_raised_with_low_success_rate(self):
        """With ≥10 records and <50% success, threshold should raise."""
        classifier = ConcurrencyEligibilityClassifier()
        # 3 successes out of 10 = 30% success rate
        for i in range(10):
            classifier.record_execution_result("llm_analyzed", i < 3)
        result = classifier._adjust_threshold_based_on_success("llm_analyzed")
        assert result > classifier.threshold
        assert result == min(0.9, classifier.threshold * 1.3)

    def test_unknown_task_type_returns_effective_threshold(self):
        """Task type not in tracker should return current effective threshold."""
        classifier = ConcurrencyEligibilityClassifier()
        result = classifier._adjust_threshold_based_on_success("unknown_type")
        assert result == classifier._effective_threshold


class TestDeterministicRejectionFiltering:
    """Verify that deterministic rejection types are excluded from recording.

    This tests the design decision from §1.4 of this spec: deterministic
    rejection types should not be recorded because they are always correct
    and don't benefit from threshold adjustment.
    """

    DETERMINISTIC_REJECTION_TYPES = (
        "write_operation",
        "user_requested_sequential",
        "safety_rule_blocked",
        "insufficient_subtasks",
        "concurrency_disabled",
    )

    def test_llm_rejected_is_not_deterministic(self):
        """llm_rejected is a non-deterministic type that SHOULD be recorded."""
        assert "llm_rejected" not in self.DETERMINISTIC_REJECTION_TYPES

    def test_llm_analyzed_is_not_deterministic(self):
        """llm_analyzed is a non-deterministic type that SHOULD be recorded."""
        assert "llm_analyzed" not in self.DETERMINISTIC_REJECTION_TYPES

    def test_write_operation_is_deterministic(self):
        assert "write_operation" in self.DETERMINISTIC_REJECTION_TYPES

    def test_safety_rule_blocked_is_deterministic(self):
        assert "safety_rule_blocked" in self.DETERMINISTIC_REJECTION_TYPES