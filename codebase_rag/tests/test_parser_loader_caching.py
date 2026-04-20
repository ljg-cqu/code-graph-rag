"""
Test suite for parser loader caching behavior.
"""

import threading

import codebase_rag.parser_loader as parser_loader_module
from codebase_rag.parser_loader import clear_parser_cache, load_parsers


class TestParserLoaderCaching:
    def test_cache_hit_on_second_call(self):
        clear_parser_cache()
        parsers1, queries1 = load_parsers()
        parsers2, queries2 = load_parsers()

        assert parsers1 is parsers2
        assert queries1 is queries2

    def test_force_reload_bypasses_cache(self):
        clear_parser_cache()
        parsers1, queries1 = load_parsers()
        parsers2, queries2 = load_parsers(force_reload=True)

        assert parsers1 is not parsers2
        assert queries1 is not queries2

    def test_clear_parser_cache_clears_cache(self):
        clear_parser_cache()
        load_parsers()

        assert parser_loader_module._cached_parsers is not None
        assert parser_loader_module._cached_queries is not None

        clear_parser_cache()

        assert parser_loader_module._cached_parsers is None
        assert parser_loader_module._cached_queries is None

    def test_thread_safety_with_concurrent_calls(self):
        clear_parser_cache()
        results: list[tuple[object, object]] = []

        def load_and_store() -> None:
            p, q = load_parsers()
            results.append((p, q))

        threads = [threading.Thread(target=load_and_store) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All threads should get the same cached objects
        first_p, first_q = results[0]
        for p, q in results[1:]:
            assert p is first_p
            assert q is first_q
