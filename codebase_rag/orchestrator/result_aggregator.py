"""
Result Aggregator module for parallel sub-agent execution.
Collects, deduplicates, and consolidates results from multiple sub-agents.
"""

import threading
from collections import defaultdict
from typing import Any

from loguru import logger

from codebase_rag.config import settings


class ResultAggregator:
    """
    Aggregates results from multiple parallel sub-agents.
    Handles deduplication, result consolidation, and output formatting.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.results: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {
            "total_subtasks": 0,
            "completed_subtasks": 0,
            "failed_subtasks": 0,
            "total_execution_time": 0,
        }

    def add_result(
        self, subtask: dict[str, Any], result: Any, execution_time: float = 0.0
    ):
        """
        Add a result from a completed sub-agent task.

        Args:
            subtask: Original subtask definition
            result: Result returned by the sub-agent
            execution_time: Time taken to execute the subtask in seconds
        """
        with self._lock:
            self.results.append(
                {"subtask": subtask, "result": result, "execution_time": execution_time}
            )
            self.metadata["completed_subtasks"] += 1
        logger.debug(f"Added result for subtask {subtask['id']}")

    def add_error(
        self, subtask: dict[str, Any], error: str, execution_time: float = 0.0
    ):
        """
        Add an error from a failed sub-agent task.

        Args:
            subtask: Original subtask definition
            error: Error message
            execution_time: Time taken before failure in seconds
        """
        with self._lock:
            self.errors.append(
                {"subtask": subtask, "error": error, "execution_time": execution_time}
            )
            self.metadata["failed_subtasks"] += 1
        logger.warning(f"Added error for subtask {subtask['id']}: {error}")

    def set_total_subtasks(self, count: int):
        """
        Set the total number of subtasks for progress tracking.

        Args:
            count: Total number of subtasks
        """
        with self._lock:
            self.metadata["total_subtasks"] = count

    def set_total_execution_time(self, time: float):
        """
        Set the total execution time for all subtasks.

        Args:
            time: Total execution time in seconds
        """
        with self._lock:
            self.metadata["total_execution_time"] = time

    def _deduplicate_results(self) -> list[dict[str, Any]]:
        """
        Deduplicate results to remove duplicate findings across subtasks if enabled.

        Returns:
            Deduplicated list of results if enabled, otherwise full list
        """
        if not settings.CGR_AGGREGATION_DEDUPLICATION_ENABLED:
            return self.results

        # Deduplicate based on result content hash
        unique_results = []
        seen_contents = set()

        for result_entry in self.results:
            result = result_entry["result"]
            content_hash = hash(str(result))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result_entry)

        duplicates_removed = len(self.results) - len(unique_results)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate results")

        return unique_results

    def _resolve_conflicts(
        self, results: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Resolve conflicting results using priority rules:
        1. Higher confidence results > lower confidence results
        2. Source priority: code > docs > LLM generation > other
        3. Flag unresolved conflicts for further processing

        Args:
            results: List of deduplicated results

        Returns:
            Tuple of (resolved results, list of unresolved conflicts)
        """
        # Source priority score mapping
        source_priority = {
            "code": 4,
            "doc": 3,
            "documentation": 3,
            "llm": 2,
            "generated": 2,
        }

        # Group results by content topic/entity for conflict detection
        grouped = defaultdict(list)
        for entry in results:
            # Extract target entity from subtask if available
            target = entry["subtask"].get(
                "target_entity", entry["subtask"].get("relative_path", "unknown")
            )
            grouped[target].append(entry)

        resolved = []
        unresolved = []

        for target, entries in grouped.items():
            if len(entries) == 1:
                resolved.append(entries[0])
                continue

            # Sort entries by priority (highest first)
            def entry_priority(entry):
                confidence = entry.get("confidence", 0.5)
                source = entry.get("source_type", "llm").lower()
                source_score = source_priority.get(source, 1)
                return (confidence, source_score)

            entries_sorted = sorted(entries, key=entry_priority, reverse=True)
            top = entries_sorted[0]
            next_top = entries_sorted[1]

            # Check if top result is clearly better
            if top.get("confidence", 0.5) >= next_top.get(
                "confidence", 0.5
            ) + 0.2 or source_priority.get(
                top.get("source_type", "llm").lower(), 1
            ) > source_priority.get(next_top.get("source_type", "llm").lower(), 1):
                resolved.append(top)
                if settings.CGR_PARALLEL_METRICS_ENABLED:
                    logger.debug(
                        f"Resolved conflict for {target}: selected {top.get('source_type', 'llm')} result with confidence {top.get('confidence', 0.5)}"
                    )
            else:
                # Unresolved conflict, flag it
                conflict_msg = f"Unresolved conflict for {target}: multiple conflicting results with similar confidence/source priority"
                unresolved.append(conflict_msg)
                resolved.append(top)
                if settings.CGR_PARALLEL_METRICS_ENABLED:
                    logger.warning(conflict_msg)

        return resolved, unresolved

    def consolidate(self, output_format: str = "markdown") -> Any:
        """
        Consolidate all results into a single output.

        Args:
            output_format: Format for output (markdown, json, text)

        Returns:
            Consolidated output in the requested format
        """
        deduplicated = self._deduplicate_results()
        resolved_results, unresolved_conflicts = self._resolve_conflicts(deduplicated)
        self.metadata["unresolved_conflicts"] = len(unresolved_conflicts)

        if output_format == "json":
            return self._format_json(resolved_results, unresolved_conflicts)
        elif output_format == "markdown":
            return self._format_markdown(resolved_results, unresolved_conflicts)
        else:
            return self._format_text(resolved_results, unresolved_conflicts)

    def _format_json(
        self, resolved_results: list[dict[str, Any]], unresolved_conflicts: list[str]
    ) -> dict[str, Any]:
        """
        Format results as structured JSON.

        Args:
            resolved_results: List of resolved deduplicated result entries
            unresolved_conflicts: List of unresolved conflict messages

        Returns:
            JSON-serializable result structure
        """
        return {
            "metadata": self.metadata,
            "results": [
                {
                    "subtask_id": entry["subtask"]["id"],
                    "file_path": entry["subtask"].get("relative_path"),
                    "result": entry["result"],
                    "execution_time": entry["execution_time"],
                    "confidence": entry.get("confidence", 1.0),
                    "source_type": entry.get("source_type", "llm"),
                }
                for entry in resolved_results
            ],
            "errors": [
                {
                    "subtask_id": entry["subtask"]["id"],
                    "file_path": entry["subtask"].get("relative_path"),
                    "error": entry["error"],
                    "execution_time": entry["execution_time"],
                }
                for entry in self.errors
            ],
            "unresolved_conflicts": unresolved_conflicts,
        }

    def _format_markdown(
        self, resolved_results: list[dict[str, Any]], unresolved_conflicts: list[str]
    ) -> str:
        """
        Format results as markdown report.

        Args:
            resolved_results: List of resolved deduplicated result entries
            unresolved_conflicts: List of unresolved conflict messages

        Returns:
            Markdown-formatted report
        """
        lines = []

        # Summary header
        lines.append("# Parallel Execution Summary")
        lines.append("")
        lines.append(f"Total subtasks: {self.metadata['total_subtasks']}")
        lines.append(f"Completed: {self.metadata['completed_subtasks']}")
        lines.append(f"Failed: {self.metadata['failed_subtasks']}")
        lines.append(f"Unresolved conflicts: {self.metadata['unresolved_conflicts']}")
        lines.append(
            f"Total execution time: {self.metadata['total_execution_time']:.2f}s"
        )
        lines.append("")

        # Unresolved conflicts section
        if unresolved_conflicts:
            lines.append("## ⚠️ Unresolved Conflicts")
            lines.append("")
            for conflict in unresolved_conflicts:
                lines.append(f"- {conflict}")
            lines.append("")

        # Results section
        if resolved_results:
            lines.append("## Results")
            lines.append("")

            # Group results by file for file-based tasks
            results_by_file = defaultdict(list)
            for entry in resolved_results:
                file_path = entry["subtask"].get("relative_path", "Unknown file")
                results_by_file[file_path].append(entry)

            for file_path, entries in results_by_file.items():
                lines.append(f"### {file_path}")
                lines.append("")
                for entry in entries:
                    result = entry["result"]
                    if isinstance(result, str):
                        lines.append(result)
                    else:
                        lines.append(str(result))
                    lines.append("")

        # Errors section
        if self.errors:
            lines.append("## Errors")
            lines.append("")
            for error_entry in self.errors:
                file_path = error_entry["subtask"].get("relative_path", "Unknown file")
                lines.append(f"### {file_path}: {error_entry['error']}")
                lines.append("")

        return "\n".join(lines)

    def _format_text(
        self, resolved_results: list[dict[str, Any]], unresolved_conflicts: list[str]
    ) -> str:
        """
        Format results as plain text.

        Args:
            resolved_results: List of resolved deduplicated result entries
            unresolved_conflicts: List of unresolved conflict messages

        Returns:
            Plain text output
        """
        lines = []

        lines.append("=== PARALLEL EXECUTION SUMMARY ===")
        lines.append(f"Total subtasks: {self.metadata['total_subtasks']}")
        lines.append(f"Completed: {self.metadata['completed_subtasks']}")
        lines.append(f"Failed: {self.metadata['failed_subtasks']}")
        lines.append(f"Unresolved conflicts: {self.metadata['unresolved_conflicts']}")
        lines.append(f"Total time: {self.metadata['total_execution_time']:.2f}s")
        lines.append("")

        if unresolved_conflicts:
            lines.append("=== WARNING: UNRESOLVED CONFLICTS ===")
            for conflict in unresolved_conflicts:
                lines.append(f"- {conflict}")
            lines.append("")

        if resolved_results:
            lines.append("=== RESULTS ===")
            for entry in resolved_results:
                file_path = entry["subtask"].get("relative_path", "Unknown file")
                lines.append(f"\n--- {file_path} ---")
                lines.append(str(entry["result"]))

        if self.errors:
            lines.append("\n=== ERRORS ===")
            for error_entry in self.errors:
                file_path = error_entry["subtask"].get("relative_path", "Unknown file")
                lines.append(f"{file_path}: {error_entry['error']}")

        return "\n".join(lines)

    def get_progress_summary(self) -> dict[str, Any]:
        """
        Get current progress summary for in-execution updates.

        Returns:
            Progress summary dictionary
        """
        total = self.metadata["total_subtasks"]
        completed = self.metadata["completed_subtasks"]
        failed = self.metadata["failed_subtasks"]

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": total - completed - failed,
            "progress_pct": (completed / total) * 100 if total > 0 else 0,
        }
