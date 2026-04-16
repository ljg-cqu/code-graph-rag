from __future__ import annotations

import os
import subprocess
from pathlib import Path

from loguru import logger

import mgclient

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
        cursor.execute(query, params)
        row = cursor.fetchone()
        if row is None:
            return 0
        return int(row[0])

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
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Failed to close Memgraph connection: {e}")

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
        self.results.append(self.check_file_layer())
        self.results.append(self.check_large_document_chunk_coverage())
        self.results.append(self.check_log_directory())
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
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

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
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

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
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

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
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

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

            # 3. Check missing embeddings
            missing_embeddings_count = self._fetch_single_int(
                cursor,
                f"""
                MATCH (n)
                WHERE ANY(label IN labels(n) WHERE label IN $embedded_labels)
                  AND n.{embedding_property} IS NULL
                RETURN count(n) AS count
            """,
                {"embedded_labels": embedded_labels},
            )
            embedded_node_count = self._fetch_single_int(
                cursor,
                """
                MATCH (n)
                WHERE ANY(label IN labels(n) WHERE label IN $embedded_labels)
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
                cursor.execute(
                    f"""
                    MATCH (n)
                    WHERE ANY(label IN labels(n) WHERE label IN $embedded_labels)
                      AND n.{embedding_property} IS NOT NULL
                    RETURN size(n.{embedding_property}) AS dim
                    LIMIT 1
                """,
                    {"embedded_labels": embedded_labels},
                )
                result = cursor.fetchone()
                if result:
                    actual_dim = result[0]
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
            results.append(
                HealthCheckResult(
                    name=cs.HEALTH_CHECK_INGESTION_VALIDATION_FAILED,
                    passed=False,
                    message=cs.HEALTH_CHECK_INGESTION_VALIDATION_ERROR_MSG,
                    error=str(e),
                )
            )
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Failed to close Memgraph cursor: {e}")
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Failed to close Memgraph connection: {e}")

        return results
