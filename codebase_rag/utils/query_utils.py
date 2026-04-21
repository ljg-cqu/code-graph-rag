"""Query utility functions (deprecated module).

This module previously contained keyword extraction functions that have been
removed per the LLM-First Orchestration spec. Use LLMQueryPlanner.expected_entities
for LLM-extracted entities instead.

This file is kept for backward compatibility but is effectively empty.
All functions have been removed as they relied on regex-based keyword matching
which is replaced by LLM-driven entity extraction.
"""

# NOTE: extract_best_keyword and extract_keywords have been removed.
# Use LLMQueryPlanner.expected_entities for LLM-extracted entities instead.

__all__ = []
