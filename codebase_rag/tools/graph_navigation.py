from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from pydantic_ai import Tool

from .. import logs as ls
from .. import tool_errors as te
from ..cypher_queries import (
    CYPHER_FIND_CALLERS,
    CYPHER_FIND_IMPORTERS,
    CYPHER_FIND_IMPLEMENTATIONS,
    CYPHER_PROJECT_STRUCTURE,
)
from ..services import QueryProtocol
from . import tool_descriptions as td

# Max depth cap for call hierarchy traversal
_MAX_DEPTH = 5


class GraphNavigator:
    __slots__ = ("project_root", "ingestor")

    def __init__(self, project_root: str, ingestor: QueryProtocol):
        self.project_root = Path(project_root).resolve()
        if self.project_root.is_file():
            self.project_root = self.project_root.parent
        self.ingestor = ingestor
        logger.info(ls.GRAPH_NAVIGATOR_INIT.format(root=self.project_root))

    async def find_references(
        self, qualified_name: str, reference_type: str = "all"
    ) -> str:
        logger.info(ls.GRAPH_NAVIGATOR_SEARCH.format(name=qualified_name))

        if reference_type not in ("calls", "imports", "all"):
            return te.ERROR_WRAPPER.format(
                message=f"Invalid reference_type: '{reference_type}'. Use 'calls', 'imports', or 'all'."
            )

        try:
            results: list[dict] = []

            if reference_type in ("calls", "all"):
                callers = await asyncio.to_thread(
                    self.ingestor.fetch_all,
                    CYPHER_FIND_CALLERS,
                    {"qn": qualified_name},
                )
                for row in callers:
                    row["ref_type"] = "calls"
                    results.append(row)

            if reference_type in ("imports", "all"):
                importers = await asyncio.to_thread(
                    self.ingestor.fetch_all,
                    CYPHER_FIND_IMPORTERS,
                    {"qn": qualified_name},
                )
                for row in importers:
                    row["ref_type"] = "imports"
                    results.append(row)

            if not results:
                return f"No references found for '{qualified_name}'."

            logger.info(
                ls.GRAPH_NAVIGATOR_FOUND.format(count=len(results), name=qualified_name)
            )

            lines = [f"References to '{qualified_name}' ({len(results)}):"]
            for i, row in enumerate(results, 1):
                ref_type = row.get("ref_type", "unknown")
                qn = row.get("qualified_name", "unknown")
                path = row.get("path", "")
                start_line = row.get("start_line")
                type_label = row.get("type", "")
                if isinstance(type_label, list):
                    type_label = type_label[0] if type_label else ""
                detail = f"  {i}. [{ref_type}] {qn}"
                if type_label:
                    detail += f" ({type_label})"
                if path:
                    if start_line:
                        detail += f" — {path}:{start_line}"
                    else:
                        detail += f" — {path}"
                lines.append(detail)

            return "\n".join(lines)

        except Exception as e:
            logger.error(ls.GRAPH_NAVIGATOR_ERROR.format(name=qualified_name, error=e))
            return te.ERROR_WRAPPER.format(message=str(e))

    async def get_call_hierarchy(
        self, qualified_name: str, direction: str = "both", depth: int = 2
    ) -> str:
        depth = min(max(depth, 1), _MAX_DEPTH)

        if direction not in ("callers", "callees", "both"):
            return te.ERROR_WRAPPER.format(
                message=f"Invalid direction: '{direction}'. Use 'callers', 'callees', or 'both'."
            )

        logger.info(
            ls.GRAPH_CALL_HIERARCHY.format(
                name=qualified_name, direction=direction, depth=depth
            )
        )

        try:
            lines = [f"Call hierarchy for '{qualified_name}' (depth={depth}):"]

            if direction in ("callers", "both"):
                # ENHANCED: Query with PageRank and community scoring for relevance ranking
                # NOTE: Memgraph does NOT support parameterized bounds in variable-length paths
                # (e.g., [:CALLS*1..$depth] causes "Property map matching" error).
                # Use string interpolation with validated integer to prevent injection.
                callers_query = f"""
                MATCH path = (caller:Function|Method)-[:CALLS*1..{depth}]->(target)
                WHERE target.qualified_name = $qn
                RETURN DISTINCT
                    caller.qualified_name AS qualified_name,
                    caller.name AS name,
                    length(path) AS depth,
                    COALESCE(caller.pagerank_score, 0.1) AS pagerank,
                    COALESCE(caller.community_importance, 0.0) AS community_importance,
                    caller.docstring AS docstring
                ORDER BY (pagerank * 0.7 + community_importance * 0.3) DESC, depth ASC
                LIMIT 50
                """
                callers = await asyncio.to_thread(
                    self.ingestor.fetch_all,
                    callers_query,
                    {"qn": qualified_name},
                )
                lines.append(f"\nCallers ({len(callers)}, ranked by importance):")
                if callers:
                    for row in callers:
                        d = row.get("depth", "?")
                        qn = row.get("qualified_name", "unknown")
                        pr = row.get("pagerank", 0)
                        ci = row.get("community_importance", 0)
                        importance = pr * 0.7 + ci * 0.3
                        ds = row.get("docstring", "")
                        docstring_preview = f" - {ds[:60]}..." if ds else ""
                        lines.append(
                            f"  {'  ' * (d - 1)}[depth {d}, importance: {importance:.3f}] {qn}{docstring_preview}"
                        )
                else:
                    lines.append("  (none)")

            if direction in ("callees", "both"):
                # ENHANCED: Query with PageRank and community scoring for relevance ranking
                # NOTE: Memgraph does NOT support parameterized bounds in variable-length paths
                # (e.g., [:CALLS*1..$depth] causes "Property map matching" error).
                # Use string interpolation with validated integer to prevent injection.
                callees_query = f"""
                MATCH path = (target)-[:CALLS*1..{depth}]->(callee:Function|Method)
                WHERE target.qualified_name = $qn
                RETURN DISTINCT
                    callee.qualified_name AS qualified_name,
                    callee.name AS name,
                    length(path) AS depth,
                    COALESCE(callee.pagerank_score, 0.1) AS pagerank,
                    COALESCE(callee.community_importance, 0.0) AS community_importance,
                    callee.docstring AS docstring
                ORDER BY (pagerank * 0.7 + community_importance * 0.3) DESC, depth ASC
                LIMIT 50
                """
                callees = await asyncio.to_thread(
                    self.ingestor.fetch_all,
                    callees_query,
                    {"qn": qualified_name},
                )
                lines.append(f"\nCallees ({len(callees)}, ranked by importance):")
                if callees:
                    for row in callees:
                        d = row.get("depth", "?")
                        qn = row.get("qualified_name", "unknown")
                        pr = row.get("pagerank", 0)
                        ci = row.get("community_importance", 0)
                        importance = pr * 0.7 + ci * 0.3
                        ds = row.get("docstring", "")
                        docstring_preview = f" - {ds[:60]}..." if ds else ""
                        lines.append(
                            f"  {'  ' * (d - 1)}[depth {d}, importance: {importance:.3f}] {qn}{docstring_preview}"
                        )
                else:
                    lines.append("  (none)")

            return "\n".join(lines)

        except Exception as e:
            logger.error(ls.GRAPH_NAVIGATOR_ERROR.format(name=qualified_name, error=e))
            return te.ERROR_WRAPPER.format(message=str(e))

    async def find_implementations(self, interface_name: str) -> str:
        logger.info(ls.GRAPH_IMPLEMENTATIONS.format(name=interface_name))

        try:
            results = await asyncio.to_thread(
                self.ingestor.fetch_all,
                CYPHER_FIND_IMPLEMENTATIONS,
                {"qn": interface_name},
            )

            if not results:
                return f"No implementations found for '{interface_name}'."

            lines = [f"Implementations of '{interface_name}' ({len(results)}):"]
            for i, row in enumerate(results, 1):
                qn = row.get("qualified_name", "unknown")
                rel_type = row.get("relationship_type", "")
                path = row.get("path", "")
                start_line = row.get("start_line")
                detail = f"  {i}. {qn}"
                if rel_type:
                    detail += f" [{rel_type}]"
                if path:
                    if start_line:
                        detail += f" — {path}:{start_line}"
                    else:
                        detail += f" — {path}"
                lines.append(detail)

            return "\n".join(lines)

        except Exception as e:
            logger.error(
                ls.GRAPH_NAVIGATOR_ERROR.format(name=interface_name, error=e)
            )
            return te.ERROR_WRAPPER.format(message=str(e))

    async def get_project_structure(self, project_name: str = "") -> str:
        logger.info(ls.GRAPH_PROJECT_STRUCTURE)

        try:
            # If no project name, infer from directory
            if not project_name:
                project_name = self.project_root.name

            results = await asyncio.to_thread(
                self.ingestor.fetch_all,
                CYPHER_PROJECT_STRUCTURE,
                {"project_name": project_name},
            )

            # Also gather filesystem info
            dir_counts: dict[str, dict] = {}
            for dirpath, _dirnames, filenames in os_walk_safe(self.project_root):
                rel = str(Path(dirpath).relative_to(self.project_root))
                if rel == ".":
                    rel = ""
                ext_counts: dict[str, int] = {}
                for f in filenames:
                    ext = Path(f).suffix or "(no ext)"
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                dir_counts[rel] = ext_counts

            lines = [f"Project structure for '{project_name}':"]

            # Graph metadata summary
            total_files = 0
            total_functions = 0
            total_classes = 0
            if results:
                lines.append("\nGraph metadata:")
                # Sort by directory path for consistent output
                sorted_results = sorted(results, key=lambda row: row.get("dir_path") or "")
                for row in sorted_results:
                    dir_path = row.get("dir_path") or "(root)"
                    fc = row.get("file_count", 0)
                    func_c = row.get("function_count", 0)
                    cls_c = row.get("class_count", 0)
                    total_files += fc
                    total_functions += func_c
                    total_classes += cls_c
                    if fc or func_c or cls_c:
                        lines.append(
                            f"  {dir_path}: {fc} files, {func_c} functions, {cls_c} classes"
                        )

            lines.append(
                f"\nTotals: {total_files} files, {total_functions} functions, {total_classes} classes"
            )

            # Filesystem summary
            lines.append("\nDirectory overview (top-level):")
            top_dirs = sorted(
                d for d in dir_counts if d and "/" not in d and "\\" not in d
            )
            for d in top_dirs[:20]:
                exts = dir_counts[d]
                ext_summary = ", ".join(
                    f"{ext}: {ct}" for ext, ct in sorted(exts.items())
                )
                lines.append(f"  {d}/ ({ext_summary})")

            return "\n".join(lines)

        except Exception as e:
            logger.error(ls.GRAPH_NAVIGATOR_ERROR.format(name=project_name, error=e))
            return te.ERROR_WRAPPER.format(message=str(e))

    async def get_import_dependencies(self, module_path: str, depth: int = 3) -> str:
        depth = min(max(depth, 1), _MAX_DEPTH)
        logger.info(ls.GRAPH_IMPORT_DEPS.format(path=module_path, depth=depth))

        try:
            # NOTE: Memgraph does NOT support parameterized bounds in variable-length paths.
            # Use string interpolation with validated integer to prevent injection.
            query = (
                f"MATCH path = (m:Module)-[:IMPORTS*1..{depth}]->(dep:Module) "
                "WHERE m.qualified_name = $qn OR m.path = $qn "
                "RETURN DISTINCT dep.qualified_name AS qualified_name, "
                "dep.path AS path, length(path) AS depth "
                "ORDER BY depth, dep.qualified_name"
            )
            results = await asyncio.to_thread(
                self.ingestor.fetch_all,
                query,
                {"qn": module_path},
            )

            if not results:
                return f"No import dependencies found for '{module_path}'."

            lines = [f"Import dependencies for '{module_path}' (depth={depth}):"]

            # Group by depth
            by_depth: dict[int, list[str]] = {}
            seen: set[str] = set()
            circular: list[str] = []

            for row in results:
                qn = row.get("qualified_name", "unknown")
                d = row.get("depth", 0)
                if qn in seen:
                    circular.append(qn)
                    continue
                seen.add(qn)
                by_depth.setdefault(d, []).append(qn)

            for d in sorted(by_depth):
                lines.append(f"\n  Depth {d} ({len(by_depth[d])} dependencies):")
                for qn in by_depth[d]:
                    lines.append(f"    - {qn}")

            if circular:
                lines.append(f"\n  Circular dependencies detected: {', '.join(circular)}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(ls.GRAPH_NAVIGATOR_ERROR.format(name=module_path, error=e))
            return te.ERROR_WRAPPER.format(message=str(e))


def os_walk_safe(root: Path) -> list[tuple[str, list[str], list[str]]]:
    """Safe os.walk that skips hidden dirs and common non-code dirs."""
    import os

    skip_dirs = {".git", ".hg", "__pycache__", "node_modules", ".tox", ".mypy_cache", "venv", ".venv"}
    results = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        # Prune in-place
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        results.append((dirpath, dirnames, filenames))
    return results


# --- Factory functions ---


def create_find_references_tool(navigator: GraphNavigator) -> Tool:
    async def find_references(qualified_name: str, reference_type: str = "all") -> str:
        return await navigator.find_references(qualified_name, reference_type)

    return Tool(
        function=find_references,
        name=td.AgenticToolName.FIND_REFERENCES,
        description=td.FIND_REFERENCES,
    )


def create_get_call_hierarchy_tool(navigator: GraphNavigator) -> Tool:
    async def get_call_hierarchy(
        qualified_name: str, direction: str = "both", depth: int = 2
    ) -> str:
        return await navigator.get_call_hierarchy(qualified_name, direction, depth)

    return Tool(
        function=get_call_hierarchy,
        name=td.AgenticToolName.GET_CALL_HIERARCHY,
        description=td.GET_CALL_HIERARCHY,
    )


def create_find_implementations_tool(navigator: GraphNavigator) -> Tool:
    async def find_implementations(interface_name: str) -> str:
        return await navigator.find_implementations(interface_name)

    return Tool(
        function=find_implementations,
        name=td.AgenticToolName.FIND_IMPLEMENTATIONS,
        description=td.FIND_IMPLEMENTATIONS,
    )


def create_get_project_structure_tool(navigator: GraphNavigator) -> Tool:
    async def get_project_structure() -> str:
        return await navigator.get_project_structure()

    return Tool(
        function=get_project_structure,
        name=td.AgenticToolName.GET_PROJECT_STRUCTURE,
        description=td.GET_PROJECT_STRUCTURE,
    )


def create_get_import_dependencies_tool(navigator: GraphNavigator) -> Tool:
    async def get_import_dependencies(module_path: str, depth: int = 3) -> str:
        return await navigator.get_import_dependencies(module_path, depth)

    return Tool(
        function=get_import_dependencies,
        name=td.AgenticToolName.GET_IMPORT_DEPENDENCIES,
        description=td.GET_IMPORT_DEPENDENCIES,
    )
