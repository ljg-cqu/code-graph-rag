"""Tests for AtomicBoolean thread-safe primitive."""

import threading

from codebase_rag.utils.atomic import AtomicBoolean


class TestAtomicBoolean:
    def test_initial_value_default(self) -> None:
        ab = AtomicBoolean()
        assert ab.get() is False

    def test_initial_value_true(self) -> None:
        ab = AtomicBoolean(True)
        assert ab.get() is True

    def test_set_returns_old_value(self) -> None:
        ab = AtomicBoolean(False)
        old = ab.set(True)
        assert old is False
        assert ab.get() is True

    def test_get_and_set(self) -> None:
        ab = AtomicBoolean(True)
        old = ab.get_and_set(False)
        assert old is True
        assert ab.get() is False

    def test_compare_and_set_success(self) -> None:
        ab = AtomicBoolean(False)
        result = ab.compare_and_set(False, True)
        assert result is True
        assert ab.get() is True

    def test_compare_and_set_failure(self) -> None:
        ab = AtomicBoolean(True)
        result = ab.compare_and_set(False, True)
        assert result is False
        assert ab.get() is True

    def test_concurrent_set(self) -> None:
        ab = AtomicBoolean(False)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(1000):
                    ab.get()
                    ab.set(True)
                    ab.set(False)
                    ab.compare_and_set(True, False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert ab.get() in (True, False)

    def test_only_one_cas_succeeds(self) -> None:
        ab = AtomicBoolean(False)
        success_count = 0
        lock = threading.Lock()

        def try_cas() -> None:
            nonlocal success_count
            if ab.compare_and_set(False, True):
                with lock:
                    success_count += 1

        threads = [threading.Thread(target=try_cas) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success_count == 1
