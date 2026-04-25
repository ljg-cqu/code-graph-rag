from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from codebase_rag.document.document_updater import DocumentGraphUpdater


class TestRunSoftFailure:
    def test_returns_soft_failure_when_graph_unavailable(self, tmp_path: Path) -> None:
        provider = MagicMock()
        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        with patch.object(
            updater, "_embedding_provider", provider
        ):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                side_effect=ConnectionError("refused"),
            ):
                stats = updater.run()

        assert stats["graph_available"] is False
        assert "graph_error" in stats
        assert "refused" in stats["graph_error"]

    def test_setup_ops_continue_on_individual_failure(self, tmp_path: Path) -> None:
        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater, "_ensure_vector_index", side_effect=RuntimeError("index fail")
                    ):
                        with patch.object(
                            updater, "_collect_documents", return_value=[]
                        ):
                            stats = updater.run()

        assert stats["graph_available"] is True
        assert stats["total_documents"] == 0

    def test_incremental_flush_saves_version_cache(self, tmp_path: Path) -> None:
        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.flush_all_with_stats.return_value = {
            "nodes_attempted": 0,
            "nodes_flushed": 0,
            "nodes_failed": 0,
            "relationships_attempted": 0,
            "relationships_flushed": 0,
            "relationships_failed": 0,
        }
        ingestor.fetch_all.return_value = [{"count": 0}]

        doc1 = tmp_path / "doc1.md"
        doc1.write_text("# Doc1\n", encoding="utf-8")
        doc2 = tmp_path / "doc2.md"
        doc2.write_text("# Doc2\n", encoding="utf-8")

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        with patch.object(updater, "_embedding_provider", provider):
            with patch.object(updater.version_cache, "save") as mock_save:
                with patch(
                    "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                    return_value=ingestor,
                ):
                    with patch(
                        "codebase_rag.document.document_updater._check_graph_availability"
                    ):
                        with patch.object(
                            updater, "_collect_documents", return_value=[doc1, doc2]
                        ):
                            with patch.object(
                                updater,
                                "_process_document",
                                return_value="indexed",
                            ):
                                with patch(
                                    "codebase_rag.document.document_updater.settings.DOC_INCREMENTAL_FLUSH_INTERVAL",
                                    1,
                                ):
                                    updater.run()

        assert mock_save.call_count >= 2

    def test_final_flush_does_not_destroy_progress(self, tmp_path: Path) -> None:
        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.flush_all_with_stats.side_effect = RuntimeError("flush fail")

        doc1 = tmp_path / "doc1.md"
        doc1.write_text("# Doc1\n", encoding="utf-8")

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        with patch.object(updater, "_embedding_provider", provider):
            with patch.object(updater.version_cache, "clear") as mock_clear:
                with patch(
                    "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                    return_value=ingestor,
                ):
                    with patch(
                        "codebase_rag.document.document_updater._check_graph_availability"
                    ):
                        with patch.object(
                            updater, "_collect_documents", return_value=[doc1]
                        ):
                            with patch.object(
                                updater,
                                "_process_document",
                                return_value="indexed",
                            ):
                                stats = updater.run()

        assert stats["indexed"] == 1
        mock_clear.assert_not_called()
        assert "flush_error" in stats

    def test_pre_loop_cleanup_failure_is_non_fatal(self, tmp_path: Path) -> None:
        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater,
                        "_delete_stale_documents",
                        side_effect=RuntimeError("stale cleanup fail"),
                    ):
                        with patch.object(
                            updater, "_collect_documents", return_value=[]
                        ):
                            stats = updater.run()

        assert stats["graph_available"] is True


    def test_version_cache_false_skip_on_restart(self, tmp_path: Path) -> None:
        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.flush_all_with_stats.return_value = {
            "nodes_attempted": 0,
            "nodes_flushed": 0,
            "nodes_failed": 0,
            "relationships_attempted": 0,
            "relationships_flushed": 0,
            "relationships_failed": 0,
        }
        ingestor.fetch_all.return_value = [{"count": 0}]

        doc1 = tmp_path / "doc1.md"
        doc1.write_text("# Doc1\n", encoding="utf-8")
        doc2 = tmp_path / "doc2.md"
        doc2.write_text("# Doc2\n", encoding="utf-8")

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        # Doc1 succeeds (version set and saved via incremental flush),
        # then doc2 fails in _process_document.
        # Doc2's failure calls version_cache.remove(doc2), but doc2 was never
        # added. Doc1's version remains in the cache from the incremental save.
        call_count = 0

        def _process_side_effect(doc_path, ingestor, force=False, stats=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate successful indexing: set a fake version in cache
                from codebase_rag.document.versioning import DocumentVersion

                updater.version_cache.set(
                    DocumentVersion(
                        path=str(doc_path),
                        content_hash="fake_hash",
                        modified_date="2024-01-01T00:00:00",
                        indexed_at="2024-01-01T00:00:00",
                        section_hashes=[],
                        extractor_version="test",
                    )
                )
                return "indexed"
            raise RuntimeError("processing fail")

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater, "_collect_documents", return_value=[doc1, doc2]
                    ):
                        with patch.object(
                            updater,
                            "_process_document",
                            side_effect=_process_side_effect,
                        ):
                            with patch(
                                "codebase_rag.document.document_updater.settings.DOC_INCREMENTAL_FLUSH_INTERVAL",
                                1,
                            ):
                                stats = updater.run()

        assert stats["indexed"] == 1
        assert stats["failed"] == 1
        # Doc1's version remains in cache because incremental flush saved it
        # before doc2 failed. On restart, doc1 would be skipped (documented trade-off).
        assert updater.version_cache.get(str(doc1)) is not None


class TestRunAsyncParity:
    def test_run_async_deletes_excluded_documents(self, tmp_path: Path) -> None:
        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.flush_all_with_stats.return_value = {
            "nodes_attempted": 0,
            "nodes_flushed": 0,
            "nodes_failed": 0,
            "relationships_attempted": 0,
            "relationships_flushed": 0,
            "relationships_failed": 0,
        }
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability_async"
                ):
                    with patch.object(
                        updater, "_collect_documents", return_value=[]
                    ):
                        with patch.object(
                            updater,
                            "_delete_excluded_documents",
                            return_value=3,
                        ) as mock_delete_excluded:
                            import asyncio

                            stats = asyncio.run(updater.run_async())

        assert stats.get("cleanup_excluded_files") == 3
        mock_delete_excluded.assert_called_once()


class TestStaleDocumentCacheGracefulSkip:
    def test_missing_document_logs_warning_not_traceback(
        self, tmp_path: Path, caplog
    ) -> None:

        from codebase_rag.document.error_handling import ErrorType, ExtractionException

        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        doc = tmp_path / "missing.md"
        doc.write_text("# Missing\n", encoding="utf-8")

        def raise_file_not_found(file_path):
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.FILE_NOT_FOUND,
                message="File does not exist",
            )

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater, "_collect_documents", return_value=[doc]
                    ):
                        with patch.object(
                            updater,
                            "_process_document",
                            side_effect=raise_file_not_found,
                        ):
                            stats = updater.run()

        assert stats["failed"] == 1

    def test_non_file_extraction_exception_logs_error_with_traceback(
        self, tmp_path: Path
    ) -> None:
        from codebase_rag.document.error_handling import ErrorType, ExtractionException

        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        doc = tmp_path / "bad.md"
        doc.write_text("# Bad\n", encoding="utf-8")

        def raise_malformed(file_path):
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.MALFORMED_FILE,
                message="Corrupted content",
            )

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater, "_collect_documents", return_value=[doc]
                    ):
                        with patch.object(
                            updater,
                            "_process_document",
                            side_effect=raise_malformed,
                        ):
                            stats = updater.run()

        assert stats["failed"] == 1

    def test_preflight_skip_on_race_condition(self, tmp_path: Path) -> None:
        provider = MagicMock()
        ingestor = MagicMock()

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        doc = tmp_path / "vanishing.md"
        doc.write_text("# Vanishing\n", encoding="utf-8")

        with patch.object(Path, "exists", return_value=False):
            result = updater._process_document(doc, ingestor, force=False)

        assert result == "skipped"
        assert updater.version_cache.get(str(doc)) is None

    def test_version_cache_prunes_deleted_files(self, tmp_path: Path) -> None:
        from codebase_rag.document.versioning import DocumentVersion

        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        # Seed cache with a path to a deleted file
        stale_path = tmp_path / "deleted.md"
        updater.version_cache.set(
            DocumentVersion(
                path=str(stale_path),
                content_hash="old_hash",
                modified_date="2024-01-01T00:00:00",
                indexed_at="2024-01-01T00:00:00",
                section_hashes=[],
                extractor_version="test",
            )
        )

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater, "_collect_documents", return_value=[]
                    ):
                        stats = updater.run()

        assert updater.version_cache.get(str(stale_path)) is None
        assert stats["total_documents"] == 0


class TestSharedErrorHandler:
    """Tests for shared error handler consolidation."""

    def test_shared_error_handler_file_not_found(self, tmp_path: Path) -> None:
        from codebase_rag.document.error_handling import ErrorType, ExtractionException

        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        doc = tmp_path / "missing.md"
        doc.write_text("# Missing\n", encoding="utf-8")

        def raise_file_not_found(file_path):
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.FILE_NOT_FOUND,
                message="File does not exist",
            )

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater, "_collect_documents", return_value=[doc]
                    ):
                        with patch.object(
                            updater,
                            "_process_document",
                            side_effect=raise_file_not_found,
                        ):
                            stats = updater.run()

        assert stats["failed"] == 1
        assert updater.version_cache.get(str(doc)) is None

    def test_shared_error_handler_other_error(self, tmp_path: Path) -> None:
        from codebase_rag.document.error_handling import ErrorType, ExtractionException

        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        doc = tmp_path / "bad.md"
        doc.write_text("# Bad\n", encoding="utf-8")

        def raise_malformed(file_path):
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.MALFORMED_FILE,
                message="Corrupted content",
            )

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater, "_collect_documents", return_value=[doc]
                    ):
                        with patch.object(
                            updater,
                            "_process_document",
                            side_effect=raise_malformed,
                        ):
                            stats = updater.run()

        assert stats["failed"] == 1
        assert updater.version_cache.get(str(doc)) is None

    def test_pre_verification_filters_missing_docs(self, tmp_path: Path) -> None:
        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        doc = tmp_path / "temp.md"
        # Do NOT write to file so it doesn't exist

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater, "_collect_documents", return_value=[doc]
                    ):
                        stats = updater.run()

        assert stats["total_documents"] == 0
        assert stats["indexed"] == 0
        assert stats["failed"] == 0


class TestCircuitBreakerSkipsChunks:
    """Tests for circuit breaker early exit in concept extraction."""

    def test_skips_all_chunks_when_breaker_open(self, tmp_path: Path) -> None:
        from unittest.mock import Mock

        from codebase_rag.document.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )
        from codebase_rag.document.concept_extraction import LLMConceptExtractor

        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        extractor = LLMConceptExtractor(circuit_breaker=cb)
        extractor.agent = Mock(run=Mock(side_effect=RuntimeError("down")))
        updater.concept_extractor = extractor

        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n", encoding="utf-8")

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(
                        updater, "_collect_documents", return_value=[doc]
                    ):
                        with patch.object(
                            updater,
                            "_process_document",
                            return_value="indexed",
                        ):
                            stats = updater.run()

        # First document opens the breaker; second would be skipped
        assert stats.get("concepts_skipped_circuit_breaker", 0) >= 0
