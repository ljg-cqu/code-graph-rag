#!/usr/bin/env python3
"""Example demonstrating parallel code review with 10 additional round-robin scheduled workers."""

from codebase_rag.orchestrator import (
    SubAgentOrchestrator,
    TaskSplitter,
)


def main():
    # Initialize orchestrator with base 5 workers, round-robin scheduling
    orchestrator = SubAgentOrchestrator(
        worker_count=5, scheduling_strategy="round-robin"
    )

    # Add 10 additional parallel workers as requested
    orchestrator.adjust_worker_count(adjustment=10)

    # Step 1: Split code review task into file-based subtasks
    splitter = TaskSplitter()
    prompt = "Perform a security review of this file: check for SQL injection, XSS vulnerabilities, insecure dependencies, and improper authentication checks."
    subtasks = splitter.split_task(prompt, strategy="file")


    # Step 2: Execute tasks with round-robin distribution
    orchestrator.execute_tasks(subtasks)

    # Step 3: Show results

    # Print summary

    orchestrator.shutdown()


if __name__ == "__main__":
    main()
