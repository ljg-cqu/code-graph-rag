"""Tests for ResourceTracker context manager."""

from codebase_rag.utils.resource_tracker import ResourceTracker, tracked_resources


class TestResourceTracker:
    def test_cleanup_all_calls_in_reverse_order(self) -> None:
        tracker = ResourceTracker()
        call_order: list[int] = []

        tracker.track(1, lambda x: call_order.append(x))
        tracker.track(2, lambda x: call_order.append(x))
        tracker.track(3, lambda x: call_order.append(x))
        tracker.cleanup_all()

        assert call_order == [3, 2, 1]

    def test_cleanup_handles_exceptions(self) -> None:
        tracker = ResourceTracker()
        cleaned: list[int] = []

        def bad_cleanup(x: int) -> None:
            raise RuntimeError("boom")

        tracker.track(1, bad_cleanup)
        tracker.track(2, lambda x: cleaned.append(x))
        tracker.cleanup_all()

        assert cleaned == [2]

    def test_cleanup_idempotent(self) -> None:
        tracker = ResourceTracker()
        cleaned: list[int] = []
        tracker.track(1, lambda x: cleaned.append(x))

        tracker.cleanup_all()
        tracker.cleanup_all()

        assert cleaned == [1]

    def test_tracked_resources_context_manager(self) -> None:
        cleaned: list[int] = []

        with tracked_resources() as tracker:
            tracker.track(1, lambda x: cleaned.append(x))
            tracker.track(2, lambda x: cleaned.append(x))

        assert cleaned == [2, 1]

    def test_tracked_resources_cleans_up_on_exception(self) -> None:
        cleaned: list[int] = []

        try:
            with tracked_resources() as tracker:
                tracker.track(1, lambda x: cleaned.append(x))
                raise ValueError("test")
        except ValueError:
            pass

        assert cleaned == [1]
