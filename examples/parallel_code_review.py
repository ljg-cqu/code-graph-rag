#!/usr/bin/env python3
"""Example demonstrating parallel code review using sub-agents."""

from codebase_rag.orchestrator import (
    SubAgentOrchestrator,
    TaskSplitter,
)


def main():
    # Step 1: Split the code review task into file-based subtasks
    splitter = TaskSplitter()
    prompt = "Review this Python file for security vulnerabilities, performance issues, and code style problems."
    subtasks = splitter.split_task(prompt, strategy="file")


    # Validate subtasks cover all code files
    if not splitter.validate_subtasks(subtasks, prompt):
        return

    # Step 2: Initialize sub-agent orchestrator with 10 parallel workers (per optimization spec)
    orchestrator = SubAgentOrchestrator(worker_count=10)

    # Step 3: Execute subtasks in parallel
    orchestrator.execute_tasks(subtasks)

    # Step 4: Consolidate and print results


    # Cleanup
    orchestrator.shutdown()


if __name__ == "__main__":
    main()
