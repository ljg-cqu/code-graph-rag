"""
Result Aggregator module for parallel sub-agent execution.
Collects, deduplicates, and consolidates results from multiple sub-agents.
"""

from collections import defaultdict
from typing import Any

from loguru import logger


class ResultAggregator:
    """
    Aggregates results from multiple parallel sub-agents.
    Handles deduplication, result consolidation, and output formatting.
    """

    def __init__(self):
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
        self.metadata["total_subtasks"] = count

    def set_total_execution_time(self, time: float):
        """
        Set the total execution time for all subtasks.

        Args:
            time: Total execution time in seconds
        """
        self.metadata["total_execution_time"] = time

    def _deduplicate_results(self) -> list[dict[str, Any]]:
        """
        Deduplicate results to remove duplicate findings across subtasks.

        Returns:
            Deduplicated list of results
        """
        # For MVP, simple deduplication based on result content hash
        unique_results = []
        seen_contents = set()

        for result_entry in self.results:
            result = result_entry["result"]
            # Convert result to string for hashing (works for simple types)
            content_hash = hash(str(result))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result_entry)

        duplicates_removed = len(self.results) - len(unique_results)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate results")

        return unique_results

    def consolidate(self, output_format: str = "markdown") -> Any:
        """
        Consolidate all results into a single output.

        Args:
            output_format: Format for output (markdown, json, text)

        Returns:
            Consolidated output in the requested format
        """
        deduplicated = self._deduplicate_results()

        if output_format == "json":
            return self._format_json(deduplicated)
        elif output_format == "markdown":
            return self._format_markdown(deduplicated)
        else:
            return self._format_text(deduplicated)

    def _format_json(self, deduplicated: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Format results as structured JSON.

        Args:
            deduplicated: List of deduplicated result entries

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
                }
                for entry in deduplicated
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
        }

    def _format_markdown(self, deduplicated: list[dict[str, Any]]) -> str:
        """
        Format results as markdown report.

        Args:
            deduplicated: List of deduplicated result entries

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
        lines.append(
            f"Total execution time: {self.metadata['total_execution_time']:.2f}s"
        )
        lines.append("")

        # Results section
        if deduplicated:
            lines.append("## Results")
            lines.append("")

            # Group results by file for file-based tasks
            results_by_file = defaultdict(list)
            for entry in deduplicated:
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

    def _format_text(self, deduplicated: list[dict[str, Any]]) -> str:
        """
        Format results as plain text.

        Args:
            deduplicated: List of deduplicated result entries

        Returns:
            Plain text output
        """
        lines = []

        lines.append("=== PARALLEL EXECUTION SUMMARY ===")
        lines.append(f"Total subtasks: {self.metadata['total_subtasks']}")
        lines.append(f"Completed: {self.metadata['completed_subtasks']}")
        lines.append(f"Failed: {self.metadata['failed_subtasks']}")
        lines.append(f"Total time: {self.metadata['total_execution_time']:.2f}s")
        lines.append("")

        if deduplicated:
            lines.append("=== RESULTS ===")
            for entry in deduplicated:
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
