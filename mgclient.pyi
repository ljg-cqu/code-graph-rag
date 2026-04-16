from __future__ import annotations

from typing import Any

class ColumnDescriptor:
    name: str


class Cursor:
    description: list[ColumnDescriptor] | None

    def execute(self, query: str, params: Any = ...) -> None: ...
    def fetchall(self) -> list[tuple[Any, ...]]: ...
    def fetchone(self) -> tuple[Any, ...] | None: ...
    def close(self) -> None: ...


class Connection:
    autocommit: bool

    def cursor(self) -> Cursor: ...
    def close(self) -> None: ...


class MemgraphError(Exception): ...


def connect(
    *,
    host: str,
    port: int,
    username: str | None = ...,
    password: str | None = ...,
    client_name: str | None = ...,
    lazy: bool | None = ...,
    sslmode: Any = ...,
) -> Connection: ...
