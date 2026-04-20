from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from loguru import logger

import mgclient
from ..exceptions import QueryExecutionError
from ..services.graph_service import MemgraphIngestor

from .. import constants as cs
from ..config import settings
from ..schemas import HealthCheckResult


class HealthChecker:
    __slots__ = ("results",)

    def __init__(self):
        self.results: list[HealthCheckResult] = []

    @staticmethod
    def _parse_label_expression(label_expression: str) -> list[str]:
        labels = [
            label.strip() for label in label_expression.split("|") if label.strip()
        ]
        return labels or [label_expression]

    @staticmethod
    def _fetch_single_int(
        cursor: mgclient.Cursor,
        query: str,
        params: dict[str, object] | None = None,
    ) -> int:
        try:
            cursor.execute(query, params)
            row = cursor.fetchone()
            # Consume any remaining rows to allow safe connection close
            HealthChecker._consume_all_results(cursor)
            if row is None:
                return 0
            return int(row[0])
        except Exception as e:
            raise QueryExecutionError(query, params, e) from e

    def _fetch_embedding_dim(
        self,
        cursor: mgclient.Cursor,
        embedded_labels: list[str],
        embedding_property: str,
    ) -> int | None:
        """Fetch embedding dimension with error handling."""
        query = f"""
            MATCH (n)
            WHERE ANY(label IN labels(n) WHERE label IN $embedded_labels)
              AND n.{embedding_property} IS NOT NULL
            RETURN size(n.{embedding_property}) AS dim
            LIMIT 1
        """
        params = {"embedded_labels": embedded_labels}

        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            HealthChecker._consume_all_results(cursor)
            return result[0] if result else None
        except Exception as e:
            raise QueryExecutionError(query, params, e) from e

    @staticmethod
    def _consume_all_results(cursor: mgclient.Cursor) -> None:
        """Consume all pending results to allow safe connection close."""
        try:
            while cursor.fetchone() is not None:
                pass
        except Exception:
            pass  # Ignore errors during cleanup

    def check_docker(self) -> HealthCheckResult:
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_DOCKER_RUNNING,
                    passed=True,
                    message=cs.HEALTH_CHECK_DOCKER_RUNNING_MSG.format(version=version),
                )
            else:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_DOCKER_NOT_RUNNING,
                    passed=False,
                    message=cs.HEALTH_CHECK_DOCKER_NOT_RESPONDING_MSG,
                    error=result.stderr.strip() or cs.HEALTH_CHECK_DOCKER_EXIT_CODE,
                )
        except FileNotFoundError:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_DOCKER_NOT_RUNNING,
                passed=False,
                message=cs.HEALTH_CHECK_DOCKER_NOT_INSTALLED_MSG,
                error=cs.HEALTH_CHECK_DOCKER_NOT_IN_PATH,
            )
        except subprocess.TimeoutExpired:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_DOCKER_NOT_RUNNING,
                passed=False,
                message=cs.HEALTH_CHECK_DOCKER_TIMEOUT_MSG,
                error=cs.HEALTH_CHECK_DOCKER_TIMEOUT_ERROR,
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_DOCKER_NOT_RUNNING,
                passed=False,
                message=cs.HEALTH_CHECK_DOCKER_FAILED_MSG,
                error=str(e),
            )

    def check_memgraph_connection(self) -> HealthCheckResult:
        conn = None
        cursor = None
        try:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )

            cursor = conn.cursor()
            cursor.execute(cs.HEALTH_CHECK_MEMGRAPH_QUERY)
            list(cursor.fetchall())

            return HealthCheckResult(
                name=cs.HEALTH_CHECK_MEMGRAPH_SUCCESSFUL,
                passed=True,
                message=cs.HEALTH_CHECK_MEMGRAPH_CONNECTED_MSG.format(
                    host=settings.MEMGRAPH_HOST,
                    port=settings.MEMGRAPH_PORT,
                ),
            )

        except mgclient.MemgraphError as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_MEMGRAPH_FAILED,
                passed=False,
                message=cs.HEALTH_CHECK_MEMGRAPH_CONNECTION_FAILED_MSG,
                error=cs.HEALTH_CHECK_MEMGRAPH_ERROR.format(error=str(e)),
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_MEMGRAPH_FAILED,
                passed=False,
                message=cs.HEALTH_CHECK_MEMGRAPH_UNEXPECTED_FAILURE_MSG,
                error=str(e),
            )
        finally:
            if cursor is not None:
                try:
                    # Consume any pending results before closing
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph connection: {e}")

    def check_api_key(self, env_name: str, display_name: str) -> HealthCheckResult:
        value = os.getenv(env_name) or getattr(settings, env_name, None)
        passed = bool(value)
        error_msg = (
            None
            if passed
            else cs.HEALTH_CHECK_API_KEY_MISSING_MSG.format(env_name=env_name)
        )
        return HealthCheckResult(
            name=(
                cs.HEALTH_CHECK_API_KEY_SET.format(display_name=display_name)
                if passed
                else cs.HEALTH_CHECK_API_KEY_NOT_SET.format(display_name=display_name)
            ),
            passed=passed,
            message=cs.HEALTH_CHECK_API_KEY_CONFIGURED
            if passed
            else cs.HEALTH_CHECK_API_KEY_NOT_CONFIGURED,
            error=error_msg,
        )

    def check_api_keys(self) -> list[HealthCheckResult]:
        return [
            self.check_api_key(env_name, display_name)
            for env_name, display_name in cs.HEALTH_CHECK_TOOLS
        ]

    def check_external_tool(
        self, tool_name: str, command: str | None = None
    ) -> HealthCheckResult:
        cmd = command or tool_name
        check_cmd = [
            cs.SHELL_CMD_WHERE if os.name == "nt" else cs.SHELL_CMD_WHICH,
            cmd,
        ]

        try:
            result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
            if result.returncode == 0:
                path = result.stdout.strip().splitlines()[0]
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_TOOL_INSTALLED.format(tool_name=tool_name),
                    passed=True,
                    message=cs.HEALTH_CHECK_TOOL_INSTALLED_MSG.format(path=path),
                )
            else:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_TOOL_NOT_INSTALLED.format(tool_name=tool_name),
                    passed=False,
                    message=cs.HEALTH_CHECK_TOOL_NOT_IN_PATH_MSG.format(cmd=cmd),
                    error=cs.HEALTH_CHECK_TOOL_NOT_IN_PATH_MSG.format(cmd=cmd),
                )
        except subprocess.TimeoutExpired:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_TOOL_NOT_INSTALLED.format(tool_name=tool_name),
                passed=False,
                message=cs.HEALTH_CHECK_TOOL_TIMEOUT_MSG,
                error=cs.HEALTH_CHECK_TOOL_TIMEOUT_ERROR.format(cmd=cmd),
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_TOOL_NOT_INSTALLED.format(tool_name=tool_name),
                passed=False,
                message=cs.HEALTH_CHECK_TOOL_FAILED_MSG,
                error=str(e),
            )

    def run_all_checks(self) -> list[HealthCheckResult]:
        self.results = []
        self.results.append(self.check_docker())
        self.results.append(self.check_memgraph_connection())
        self.results.extend(self.check_api_keys())
        for tool_name, cmd in cs.HEALTH_CHECK_EXTERNAL_TOOLS:
            self.results.append(self.check_external_tool(tool_name, cmd))
        self.results.append(self.check_disconnected_nodes())
        self.results.append(self.check_required_properties())
        self.results.append(self.check_embedding_correlation())
        self.results.append(self.check_vector_indexes())
        self.results.append(self.check_vector_search())
        self.results.append(self.check_file_layer())
        self.results.append(self.check_large_document_chunk_coverage())
        self.results.append(self.check_log_directory())
        self.results.append(self.check_data_migrations_needed())
        sample_json_path = Path(cs.HEALTH_CHECK_JSON_SAMPLE_FILE)
        if sample_json_path.exists():
            self.results.append(self.check_json_ingestion_schema(str(sample_json_path)))
        return self.results

    def get_summary(self) -> tuple[int, int]:
        passed = sum(1 for r in self.results if r.passed)
        return passed, len(self.results)

    def check_disconnected_nodes(self) -> HealthCheckResult:
        from ..graph.query_generator import QueryGenerator

        conn = None
        cursor = None
        try:
            query = QueryGenerator().get_disconnected_nodes_query()
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            count = int(row[0]) if row else 0
            if count == 0:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_DISCONNECTED_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_DISCONNECTED_PASS_MSG,
                )
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_DISCONNECTED_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_DISCONNECTED_FAIL_MSG.format(count=count),
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_DISCONNECTED_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_DISCONNECTED_FAIL,
                error=cs.HEALTH_CHECK_DISCONNECTED_ERROR_MSG.format(error=str(e)),
            )
        finally:
            if cursor is not None:
                try:
                    # Consume any pending results before closing
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph connection: {e}")

    def check_required_properties(self) -> HealthCheckResult:
        conn = None
        cursor = None
        try:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()
            cursor.execute(cs.QUERY_GEN_REQUIRED_PROPS)
            row = cursor.fetchone()
            count = int(row[0]) if row else 0
            if count == 0:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_REQUIRED_PROPS_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_REQUIRED_PROPS_PASS_MSG,
                )
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_REQUIRED_PROPS_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_REQUIRED_PROPS_FAIL_MSG.format(count=count),
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_REQUIRED_PROPS_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_REQUIRED_PROPS_FAIL,
                error=cs.HEALTH_CHECK_REQUIRED_PROPS_ERROR_MSG.format(error=str(e)),
            )
        finally:
            if cursor is not None:
                try:
                    # Consume any pending results before closing
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph connection: {e}")

    def check_embedding_correlation(self) -> HealthCheckResult:
        conn = None
        cursor = None
        try:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()
            cursor.execute(
                cs.QUERY_GEN_EMBEDDING_MODEL_MISMATCH,
                {"expected_model": settings.EMBEDDING_MODEL},
            )
            row = cursor.fetchone()
            count = int(row[0]) if row else 0
            if count == 0:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_EMBEDDING_CORR_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_EMBEDDING_CORR_PASS_MSG.format(
                        model=settings.EMBEDDING_MODEL
                    ),
                )
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_EMBEDDING_CORR_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_EMBEDDING_CORR_FAIL_MSG.format(
                    count=count, model=settings.EMBEDDING_MODEL
                ),
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_EMBEDDING_CORR_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_EMBEDDING_CORR_FAIL,
                error=cs.HEALTH_CHECK_EMBEDDING_CORR_ERROR_MSG.format(error=str(e)),
            )
        finally:
            if cursor is not None:
                try:
                    # Consume any pending results before closing
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def check_file_layer(self) -> HealthCheckResult:
        conn = None
        cursor = None
        try:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()
            cursor.execute("MATCH (m:Module) RETURN count(m) AS module_count")
            module_row = cursor.fetchone()
            module_count = int(module_row[0]) if module_row else 0
            # Consume any remaining results before next query
            HealthChecker._consume_all_results(cursor)

            cursor.execute("MATCH (f:File) RETURN count(f) AS file_count")
            file_row = cursor.fetchone()
            file_count = int(file_row[0]) if file_row else 0

            if module_count == 0:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_FILE_LAYER_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_FILE_LAYER_SKIP_MSG,
                )

            if file_count > 0:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_FILE_LAYER_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_FILE_LAYER_PASS_MSG,
                )

            return HealthCheckResult(
                name=cs.HEALTH_CHECK_FILE_LAYER_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_FILE_LAYER_FAIL_MSG.format(
                    module_count=module_count
                ),
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_FILE_LAYER_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_FILE_LAYER_FAIL,
                error=cs.HEALTH_CHECK_FILE_LAYER_ERROR_MSG.format(error=str(e)),
            )
        finally:
            if cursor is not None:
                try:
                    # Consume any pending results before closing
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph connection: {e}")

    def check_large_document_chunk_coverage(self) -> HealthCheckResult:
        conn = None
        cursor = None
        try:
            conn = mgclient.connect(
                host=settings.DOC_MEMGRAPH_HOST,
                port=settings.DOC_MEMGRAPH_PORT,
            )
            cursor = conn.cursor()
            cursor.execute(
                """
                MATCH (d:Document)
                WHERE coalesce(d.word_count, 0) >= 1000
                OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
                WITH d, count(c) AS chunk_count
                WITH count(d) AS large_doc_count,
                     count(CASE WHEN chunk_count = 0 THEN 1 END) AS unchunked_count
                RETURN large_doc_count, unchunked_count
                """
            )
            row = cursor.fetchone()
            large_doc_count = int(row[0]) if row else 0
            unchunked_count = int(row[1]) if row else 0

            if large_doc_count == 0:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_DOC_CHUNK_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_DOC_CHUNK_SKIP_MSG,
                )

            if unchunked_count == 0:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_DOC_CHUNK_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_DOC_CHUNK_PASS_MSG,
                )

            return HealthCheckResult(
                name=cs.HEALTH_CHECK_DOC_CHUNK_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_DOC_CHUNK_FAIL_MSG.format(
                    count=unchunked_count
                ),
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_DOC_CHUNK_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_DOC_CHUNK_FAIL,
                error=cs.HEALTH_CHECK_DOC_CHUNK_ERROR_MSG.format(error=str(e)),
            )
        finally:
            if cursor is not None:
                try:
                    # Consume any pending results before closing
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph connection: {e}")

    def check_json_ingestion_schema(self, json_path: str) -> HealthCheckResult:
        import json

        import jsonschema

        try:
            with open(cs.HEALTH_CHECK_JSON_SCHEMA_FILE) as f:
                schema = json.load(f)
            with open(json_path) as f:
                data = json.load(f)
            try:
                jsonschema.validate(instance=data, schema=schema)
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_JSON_SCHEMA_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_JSON_SCHEMA_PASS_MSG,
                )
            except jsonschema.ValidationError as e:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_JSON_SCHEMA_FAIL,
                    passed=False,
                    message=cs.HEALTH_CHECK_JSON_SCHEMA_FAIL_MSG.format(
                        error=e.message
                    ),
                )
        except OSError as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_JSON_SCHEMA_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_JSON_SCHEMA_FAIL,
                error=cs.HEALTH_CHECK_JSON_SCHEMA_IO_ERROR_MSG.format(error=str(e)),
            )

    def check_log_directory(self) -> HealthCheckResult:
        """Check if log directory exists and is writable"""
        return HealthCheckResult(
            name="Log directory check skipped",
            passed=True,
            message="File logging is disabled, all logs go to terminal"
        )

    def check_data_migrations_needed(self) -> HealthCheckResult:
        """Check if data migrations are needed."""
        conn = None
        cursor = None
        issues = []

        try:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()

            cursor.execute("""
                MATCH (f:Function)
                WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-()
                RETURN count(f)
            """)
            orphaned_builtins = cursor.fetchone()[0]
            if orphaned_builtins > 0:
                issues.append(f"{orphaned_builtins} orphaned builtin functions")

            cursor.execute("""
                MATCH (m:Module)
                WHERE m.is_external = true AND m.path IS NOT NULL
                RETURN count(m)
            """)
            external_with_path = cursor.fetchone()[0]
            if external_with_path > 0:
                issues.append(f"{external_with_path} external modules with path set")

            cursor.execute("""
                MATCH (n)
                WHERE any(label IN labels(n) WHERE label IN ['JsonObject', 'JsonArray', 'JsonField', 'JsonValue'])
                AND n.name IS NULL
                RETURN count(n)
            """)
            json_missing_name = cursor.fetchone()[0]
            if json_missing_name > 0:
                issues.append(f"{json_missing_name} JSON nodes missing name")

            cursor.execute("""
                MATCH (t:Test)
                WHERE t.name IS NULL AND t.qualified_name IS NULL
                RETURN count(t)
            """)
            incomplete_tests = cursor.fetchone()[0]
            if incomplete_tests > 0:
                issues.append(f"{incomplete_tests} incomplete Test nodes")

            if not issues:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_MIGRATION_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_MIGRATION_PASS_MSG,
                )

            return HealthCheckResult(
                name=cs.HEALTH_CHECK_MIGRATION_NEEDED,
                passed=False,
                message=cs.HEALTH_CHECK_MIGRATION_NEEDED_MSG.format(issues="; ".join(issues)),
                error=cs.HEALTH_CHECK_MIGRATION_ERROR_MSG,
            )

        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_MIGRATION_NEEDED,
                passed=False,
                message=cs.HEALTH_CHECK_MIGRATION_NEEDED,
                error=str(e),
            )
        finally:
            if cursor is not None:
                try:
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph connection: {e}")

    def check_vector_indexes(self) -> HealthCheckResult:
        """Check if vector indexes exist for embeddable node types."""
        from ..vector_store_memgraph import MemgraphBackend

        conn = None
        cursor = None
        try:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()

            # Get existing vector indexes
            cursor.execute("SHOW VECTOR INDEX INFO;")
            existing_indexes = {row[0] for row in cursor.fetchall()}

            # Check which expected indexes are missing
            expected_labels = MemgraphBackend.LABELS_TO_INDEX
            missing_indexes = []
            for label in expected_labels:
                index_name = f"{label.lower()}_embedding_index"
                if index_name not in existing_indexes:
                    missing_indexes.append(index_name)

            if not missing_indexes:
                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_VECTOR_INDEX_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_VECTOR_INDEX_PASS_MSG,
                )

            return HealthCheckResult(
                name=cs.HEALTH_CHECK_VECTOR_INDEX_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_VECTOR_INDEX_FAIL_MSG.format(
                    missing_count=len(missing_indexes)
                ),
                error=f"Missing indexes: {', '.join(missing_indexes)}",
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_VECTOR_INDEX_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_VECTOR_INDEX_FAIL,
                error=cs.HEALTH_CHECK_VECTOR_INDEX_ERROR_MSG.format(error=str(e)),
            )
        finally:
            if cursor is not None:
                try:
                    # Consume any pending results before closing
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph connection: {e}")

    def check_vector_search(self) -> HealthCheckResult:
        """Check if vector search is functional and has embeddings."""
        from ..embeddings import get_embedding_provider
        from ..vector_backend import get_shared_backend

        try:
            # Get embedding provider
            config = settings.active_embedding_config
            provider = get_embedding_provider(config=config)

            # Generate test embedding
            test_embedding = provider.embed("test query")

            # Get vector backend and search
            backend = get_shared_backend()
            results = backend.search(test_embedding, top_k=3)

            if not results:
                # Check if any embeddings exist
                stats = backend.get_stats()
                total_embeddings = stats.get("total_embeddings", 0)

                if total_embeddings == 0:
                    return HealthCheckResult(
                        name=cs.HEALTH_CHECK_VECTOR_SEARCH_FAIL,
                        passed=False,
                        message=cs.HEALTH_CHECK_VECTOR_SEARCH_NO_EMBEDDINGS_MSG,
                        error="No embeddings stored - run 'cgr start --index-code' to index your codebase",
                    )

                return HealthCheckResult(
                    name=cs.HEALTH_CHECK_VECTOR_SEARCH_PASS,
                    passed=True,
                    message=cs.HEALTH_CHECK_VECTOR_SEARCH_PASS_MSG.format(count=total_embeddings),
                )

            # Get total embeddings for message
            stats = backend.get_stats()
            total_embeddings = stats.get("total_embeddings", 0)

            return HealthCheckResult(
                name=cs.HEALTH_CHECK_VECTOR_SEARCH_PASS,
                passed=True,
                message=cs.HEALTH_CHECK_VECTOR_SEARCH_PASS_MSG.format(count=total_embeddings),
            )
        except Exception as e:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_VECTOR_SEARCH_FAIL,
                passed=False,
                message=cs.HEALTH_CHECK_VECTOR_SEARCH_FAIL,
                error=cs.HEALTH_CHECK_VECTOR_SEARCH_ERROR_MSG.format(error=str(e)),
            )

    def validate_ingestion_quality(
        self,
        expected_node_count: int | None = None,
        expected_edge_count: int | None = None,
        node_label: str = "File",
        embedded_node_label: str = "Function",
        embedding_property: str = "embedding",
        vector_dim: int | None = None,
    ) -> list[HealthCheckResult]:
        """Run post-ingestion data quality validation checks.

        Args:
            expected_node_count: Expected number of nodes ingested (optional).
            expected_edge_count: Expected number of edges ingested (optional).
            node_label: Label of nodes to validate (default: "File").
            embedding_property: Name of embedding property on nodes (default: "embedding").
            vector_dim: Expected vector dimension (optional, auto-detected if not provided).

        Returns:
            List of HealthCheckResult objects for each validation check.
        """
        results: list[HealthCheckResult] = []
        conn = None
        cursor = None

        if vector_dim is None:
            vector_dim = settings.get_effective_vector_dim()

        embedded_labels = self._parse_label_expression(embedded_node_label)

        try:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()

            # 1. Check node count
            actual_node_count = self._fetch_single_int(
                cursor, f"MATCH (n:{node_label}) RETURN count(n) AS count"
            )
            if expected_node_count is not None:
                node_count_passed = actual_node_count == expected_node_count
                results.append(
                    HealthCheckResult(
                        name=cs.HEALTH_CHECK_NODE_COUNT,
                        passed=node_count_passed,
                        message=(
                            cs.HEALTH_CHECK_NODE_COUNT_OK_MSG.format(
                                actual=actual_node_count, expected=expected_node_count
                            )
                            if node_count_passed
                            else cs.HEALTH_CHECK_NODE_COUNT_MISMATCH_MSG.format(
                                actual=actual_node_count, expected=expected_node_count
                            )
                        ),
                        error=None
                        if node_count_passed
                        else f"Expected {expected_node_count} nodes, got {actual_node_count}",
                    )
                )
            else:
                results.append(
                    HealthCheckResult(
                        name=cs.HEALTH_CHECK_NODE_COUNT,
                        passed=True,
                        message=cs.HEALTH_CHECK_NODE_COUNT_SKIP_MSG.format(
                            count=actual_node_count
                        ),
                        error=None,
                    )
                )

            # 2. Check edge count
            actual_edge_count = self._fetch_single_int(
                cursor, "MATCH ()-->() RETURN count(*) AS count"
            )
            if expected_edge_count is not None:
                edge_count_passed = actual_edge_count == expected_edge_count
                results.append(
                    HealthCheckResult(
                        name=cs.HEALTH_CHECK_EDGE_COUNT,
                        passed=edge_count_passed,
                        message=(
                            cs.HEALTH_CHECK_EDGE_COUNT_OK_MSG.format(
                                actual=actual_edge_count, expected=expected_edge_count
                            )
                            if edge_count_passed
                            else cs.HEALTH_CHECK_EDGE_COUNT_MISMATCH_MSG.format(
                                actual=actual_edge_count, expected=expected_edge_count
                            )
                        ),
                        error=None
                        if edge_count_passed
                        else f"Expected {expected_edge_count} edges, got {actual_edge_count}",
                    )
                )
            else:
                results.append(
                    HealthCheckResult(
                        name=cs.HEALTH_CHECK_EDGE_COUNT,
                        passed=True,
                        message=cs.HEALTH_CHECK_EDGE_COUNT_SKIP_MSG.format(
                            count=actual_edge_count
                        ),
                        error=None,
                    )
                )

            # 3. Check missing embeddings (exclude builtin functions)
            missing_embeddings_count = self._fetch_single_int(
                cursor,
                f"""
                MATCH (n)
                WHERE ANY(label IN labels(n) WHERE label IN $embedded_labels)
                  AND n.{embedding_property} IS NULL
                  AND NOT (n.is_builtin = true OR n.qualified_name STARTS WITH 'builtin.')
                RETURN count(n) AS count
            """,
                {"embedded_labels": embedded_labels},
            )
            embedded_node_count = self._fetch_single_int(
                cursor,
                """
                MATCH (n)
                WHERE ANY(label IN labels(n) WHERE label IN $embedded_labels)
                  AND NOT (n.is_builtin = true OR n.qualified_name STARTS WITH 'builtin.')
                RETURN count(n) AS count
            """,
                {"embedded_labels": embedded_labels},
            )

            # Calculate allowed missing embeddings based on threshold
            if embedded_node_count == 0:
                missing_embeddings_passed = True
            else:
                missing_pct = (missing_embeddings_count / embedded_node_count) * 100
                missing_embeddings_passed = (
                    missing_pct <= settings.MAX_MISSING_EMBEDDINGS_PCT
                )
            results.append(
                HealthCheckResult(
                    name=cs.HEALTH_CHECK_MISSING_EMBEDDINGS,
                    passed=missing_embeddings_passed,
                    message=(
                        cs.HEALTH_CHECK_MISSING_EMBEDDINGS_OK_MSG
                        if missing_embeddings_passed
                        else cs.HEALTH_CHECK_MISSING_EMBEDDINGS_FOUND_MSG.format(
                            count=missing_embeddings_count
                        )
                    ),
                    error=None
                    if missing_embeddings_passed
                    else f"{missing_embeddings_count} nodes have missing embeddings",
                )
            )

            # 4. Check invalid embeddings dimension
            if missing_embeddings_count < embedded_node_count:
                actual_dim = self._fetch_embedding_dim(
                    cursor, embedded_labels, embedding_property
                )
                if actual_dim is not None:
                    dim_passed = actual_dim == vector_dim
                    results.append(
                        HealthCheckResult(
                            name=cs.HEALTH_CHECK_EMBEDDING_DIMENSION,
                            passed=dim_passed,
                            message=(
                                cs.HEALTH_CHECK_EMBEDDING_DIMENSION_OK_MSG.format(
                                    dim=actual_dim
                                )
                                if dim_passed
                                else cs.HEALTH_CHECK_EMBEDDING_DIMENSION_MISMATCH_MSG.format(
                                    actual=actual_dim, expected=vector_dim
                                )
                            ),
                            error=None
                            if dim_passed
                            else f"Expected dimension {vector_dim}, got {actual_dim}",
                        )
                    )

            # 5. Check duplicate nodes (by path)
            duplicate_count = self._fetch_single_int(cursor, f"""
                MATCH (n:{node_label})
                WITH n.path AS path, count(n) AS cnt
                WHERE cnt > 1
                RETURN count(path) AS duplicate_count
            """)
            duplicates_passed = duplicate_count == 0
            results.append(
                HealthCheckResult(
                    name=cs.HEALTH_CHECK_DUPLICATE_NODES,
                    passed=duplicates_passed,
                    message=(
                        cs.HEALTH_CHECK_DUPLICATE_NODES_OK_MSG
                        if duplicates_passed
                        else cs.HEALTH_CHECK_DUPLICATE_NODES_FOUND_MSG.format(
                            count=duplicate_count
                        )
                    ),
                    error=None
                    if duplicates_passed
                    else f"{duplicate_count} duplicate node paths found",
                )
            )

        except Exception as e:
            error_detail = str(e)
            if settings.LOG_QUALITY_CHECK_STACKTRACES:
                import traceback
                error_detail = f"{e}\n{traceback.format_exc()}"
            logger.warning(f"Quality validation error: {error_detail}")
            results.append(
                HealthCheckResult(
                    name=cs.HEALTH_CHECK_INGESTION_VALIDATION_FAILED,
                    passed=False,
                    message=cs.HEALTH_CHECK_INGESTION_VALIDATION_ERROR_MSG,
                    error=error_detail,
                )
            )
        finally:
            if cursor is not None:
                try:
                    # Consume any pending results before closing
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to close Memgraph connection: {e}")

        return results

    def get_missing_embeddings(
        self,
        embedded_labels: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, str | list[str] | None]]:
        """Get nodes missing embeddings (excludes builtin functions).

        This method identifies nodes that need embedding backfill.
        Builtin functions are excluded since they shouldn't have embeddings.

        Args:
            embedded_labels: Labels to check for embeddings (default: Function, Method).
            limit: Maximum number of results to return.

        Returns:
            List of dicts with qualified_name, labels, and path for each node.
        """
        if embedded_labels is None:
            embedded_labels = ["Function", "Method"]

        conn = None
        cursor = None
        results: list[dict[str, str | list[str] | None]] = []

        try:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()
            cursor.execute(
                cs.QUERY_GEN_MISSING_EMBEDDINGS,
                {"embedded_labels": embedded_labels, "limit": limit},
            )
            for row in cursor.fetchall():
                results.append(
                    {
                        "qualified_name": row[0],
                        "labels": list(row[1]) if row[1] else [],
                        "path": row[2],
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to get missing embeddings: {e}")
        finally:
            if cursor is not None:
                try:
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        return results

    def get_missing_embeddings_count(
        self,
        embedded_labels: list[str] | None = None,
    ) -> int:
        """Get count of nodes missing embeddings (excludes builtin functions).

        Args:
            embedded_labels: Labels to check for embeddings (default: Function, Method).

        Returns:
            Count of nodes missing embeddings.
        """
        if embedded_labels is None:
            embedded_labels = ["Function", "Method"]

        conn = None
        cursor = None

        try:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()
            cursor.execute(
                cs.QUERY_GEN_MISSING_EMBEDDINGS_COUNT,
                {"embedded_labels": embedded_labels},
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        except Exception as e:
            logger.warning(f"Failed to get missing embeddings count: {e}")
            return 0
        finally:
            if cursor is not None:
                try:
                    HealthChecker._consume_all_results(cursor)
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_runtime_status(self, ingestor: "MemgraphIngestor") -> RuntimeHealthStatus:
        """Get current runtime health status for active session.

        Call this periodically during long-running operations or after
        errors to detect capability degradation.
        """
        return get_runtime_status()



class HealthStatus(Enum):
    """Runtime health status levels."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()


@dataclass
class RuntimeHealthStatus:
    """Runtime health status for active query sessions."""

    vector_search: HealthStatus = HealthStatus.HEALTHY
    graph_traversal: HealthStatus = HealthStatus.HEALTHY
    procedures: HealthStatus = HealthStatus.HEALTHY
    last_error: str | None = None

    @property
    def overall(self) -> HealthStatus:
        """Determine overall health from individual statuses."""
        statuses = [self.vector_search, self.graph_traversal, self.procedures]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


def get_runtime_status() -> RuntimeHealthStatus:
    """Get current runtime health status for active session.

    Call this periodically during long-running operations or after
    errors to detect capability degradation.
    """
    status = RuntimeHealthStatus()

    try:
        from ..vector_backend import get_shared_backend

        backend = get_shared_backend()
        stats = backend.get_stats()
        if stats.get("total_embeddings", 0) == 0:
            status.vector_search = HealthStatus.DEGRADED
    except Exception as e:
        status.vector_search = HealthStatus.UNHEALTHY
        status.last_error = str(e)

    conn = None
    cursor = None
    try:
        conn = mgclient.connect(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
        )
        cursor = conn.cursor()
        cursor.execute("MATCH (n) RETURN count(n) LIMIT 1")
        cursor.fetchall()
    except Exception as e:
        status.graph_traversal = HealthStatus.UNHEALTHY
        status.last_error = str(e)
    finally:
        if cursor is not None:
            try:
                # Consume any pending results before closing
                HealthChecker._consume_all_results(cursor)
                cursor.close()
            except Exception as e:
                logger.debug(f"Failed to close Memgraph cursor: {e}")
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.debug(f"Failed to close Memgraph connection: {e}")

    conn = None
    cursor = None
    try:
        conn = mgclient.connect(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
        )
        cursor = conn.cursor()
        cursor.execute(
            "CALL pagerank.get() YIELD node, rank "
            "WITH rank LIMIT 1 RETURN rank"
        )
        cursor.fetchall()
    except Exception:
        status.procedures = HealthStatus.DEGRADED
    finally:
        if cursor is not None:
            try:
                # Consume any pending results before closing
                HealthChecker._consume_all_results(cursor)
                cursor.close()
            except Exception as e:
                logger.debug(f"Failed to close Memgraph cursor: {e}")
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.debug(f"Failed to close Memgraph connection: {e}")

    return status
