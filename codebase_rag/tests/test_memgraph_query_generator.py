from __future__ import annotations

from dataclasses import dataclass

from codebase_rag.graph.query_generator import MemgraphQueryGenerator


@dataclass
class _Column:
    name: str


class _FakeCursor:
    def __init__(self, handlers: dict[str, object]) -> None:
        self._handlers = handlers
        self.description: list[_Column] | None = None
        self._rows: list[tuple] = []

    def execute(self, query: str, params: dict | None = None) -> None:
        for fragment, handler in self._handlers.items():
            if fragment in query:
                if isinstance(handler, Exception):
                    raise handler
                columns, rows = handler
                self.description = [_Column(name) for name in columns]
                self._rows = rows
                return

        self.description = []
        self._rows = []

    def fetchall(self) -> list[tuple]:
        return self._rows

    def close(self) -> None:
        return None


class _FakeConnection:
    def __init__(self, handlers: dict[str, object]) -> None:
        self._handlers = handlers

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._handlers)


def test_detect_capabilities_treats_missing_test_index_as_supported() -> None:
    connection = _FakeConnection(
        {
            "SHOW VERSION": (("version",), [("3.9.0",)]),
            "CALL vector_search.search": RuntimeError(
                "Vector index 'test_index' doesn't exist."
            ),
            "SHOW INDEXES": (("name",), []),
            "CREATE INDEX IF NOT EXISTS": RuntimeError("syntax error"),
        }
    )

    capabilities = MemgraphQueryGenerator(connection).capabilities

    assert capabilities.supports_vector_search_procedure is True
    assert capabilities.supports_vector_search is True


def test_detect_capabilities_marks_missing_procedure_as_unsupported() -> None:
    connection = _FakeConnection(
        {
            "SHOW VERSION": (("version",), [("3.9.0",)]),
            "CALL vector_search.search": RuntimeError(
                "Procedure 'vector_search.search' doesn't exist."
            ),
            "RETURN cosine_similarity": RuntimeError(
                "Function 'cosine_similarity' doesn't exist."
            ),
            "RETURN vector.cosine_similarity": RuntimeError(
                "Function 'vector.cosine_similarity' doesn't exist."
            ),
            "SHOW INDEXES": (("name",), []),
            "CREATE INDEX IF NOT EXISTS": RuntimeError("syntax error"),
        }
    )

    capabilities = MemgraphQueryGenerator(connection).capabilities

    assert capabilities.supports_vector_search_procedure is False
    assert capabilities.supports_vector_search is False