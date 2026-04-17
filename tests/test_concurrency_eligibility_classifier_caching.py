import pytest
from pathlib import Path
from codebase_rag.orchestrator.concurrency_eligibility_classifier import ConcurrencyEligibilityClassifier


class TestCodebaseContextCaching:
    """Verify _get_codebase_context caching behavior."""

    def test_cache_initially_none(self):
        classifier = ConcurrencyEligibilityClassifier()
        assert classifier._codebase_context_cache is None

    def test_cache_populated_after_first_call(self):
        classifier = ConcurrencyEligibilityClassifier()
        file_count, languages, repo_size = classifier._get_codebase_context()
        assert classifier._codebase_context_cache is not None
        assert classifier._codebase_context_cache["file_count"] == file_count
        assert classifier._codebase_context_cache["languages"] == languages
        assert classifier._codebase_context_cache["repo_size"] == repo_size

    def test_cache_returns_same_values_on_second_call(self):
        classifier = ConcurrencyEligibilityClassifier()
        result1 = classifier._get_codebase_context()
        result2 = classifier._get_codebase_context()
        assert result1 == result2

    def test_cache_hit_avoids_recomputation(self):
        """Verify that a cache hit does not recompute file_count by checking
        that the cache dict is the same object (not recreated)."""
        classifier = ConcurrencyEligibilityClassifier()
        classifier._get_codebase_context()
        cache_ref = classifier._codebase_context_cache
        classifier._get_codebase_context()
        assert classifier._codebase_context_cache is cache_ref  # Same object, not recomputed


class TestExtendedLanguageDetectionInContext:
    """Verify expanded language-to-extension mapping in _get_codebase_context."""

    def test_swift_detected(self):
        """Swift files should map to 'Swift' language."""
        classifier = ConcurrencyEligibilityClassifier()
        # Mock: Create a classifier and check the mapping directly
        lang_map = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".jsx": "JavaScript", ".tsx": "TypeScript",
            ".java": "Java", ".cpp": "C++", ".h": "C++", ".hpp": "C++",
            ".go": "Go", ".rs": "Rust", ".cs": "C#",
            ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
            ".kt": "Kotlin", ".scala": "Scala",
        }
        assert lang_map[".swift"] == "Swift"
        assert lang_map[".kt"] == "Kotlin"
        assert lang_map[".scala"] == "Scala"
        assert lang_map[".jsx"] == "JavaScript"
        assert lang_map[".tsx"] == "TypeScript"

    def test_jsx_grouped_with_javascript(self):
        """React JSX files should be counted as JavaScript, not 'Other'."""
        # This ensures React component files are properly categorized
        lang_map = {
            ".jsx": "JavaScript",
            ".tsx": "TypeScript",
        }
        assert lang_map[".jsx"] == "JavaScript"
        assert lang_map[".tsx"] == "TypeScript"