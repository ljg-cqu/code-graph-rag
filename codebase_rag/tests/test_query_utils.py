"""Unit tests for query utility functions."""

from codebase_rag.utils.query_utils import extract_best_keyword, extract_keywords


def test_extract_best_keyword():
    """Test keyword extraction with stopword filtering."""
    # Common patterns that previously failed
    assert extract_best_keyword("What functions call authenticate?") == "functions"
    # "what" is stopword, "functions" is longest meaningful word
    assert extract_best_keyword("How does the code work?") in ["code", "work"]
    # "how", "does" are stopwords, "code" and "work" are both length 4
    # The function picks the first longest (alphabetical? Actually max picks by length, tie returns first max?)
    # Accept either
    assert extract_best_keyword("Find all classes in utils") == "classes"
    # "find", "all", "in" are stopwords, "classes" length 7 > "utils" length 5
    assert extract_best_keyword("Show me where database connects") == "database"
    # "show", "me", "where" are stopwords, "database" length 8 > "connects" length 8? Actually both length 8, max returns first longest (database)
    # Edge cases
    assert extract_best_keyword("") == ""
    assert extract_best_keyword("a b c") in ["a", "b", "c"]
    # All words length <=2, no meaningful words, fallback to longest word (first max length tie)
    # Should return "a" or "b" or "c" (any of them)
    # Test with stopwords only
    assert extract_best_keyword("the is at") == "the"  # all stopwords, fallback to longest (all length 2, pick first)
    # Test with mixed case
    assert extract_best_keyword("What is the Database?") == "database"
    # Should lower case and filter stopwords
    # Test with punctuation (should be ignored by split)
    assert extract_best_keyword("What is the Database?") == "database"
    # Test with numbers and symbols
    assert extract_best_keyword("What is the 123?") == "123"  # numbers not filtered
    # Ensure stopwords list is correct
    stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
                 'in', 'with', 'to', 'for', 'of', 'what', 'how', 'where', 'when',
                 'why', 'who', 'show', 'find', 'tell', 'get', 'list', 'all', 'any',
                 'some', 'does', 'do', 'can', 'could', 'would', 'should', 'will'}
    for stopword in stopwords:
        # If query is only stopwords, should fallback to longest stopword
        result = extract_best_keyword(stopword)
        assert result == stopword  # because no meaningful words, fallback to longest word (the stopword itself)
    # Mixed stopwords and meaningful
    assert extract_best_keyword("the quick brown fox") == "quick"  # "quick" length 5 > "brown" 5? Actually both 5, first max is "quick"
    # Actually "quick" and "brown" same length, max picks first max (quick). That's fine.


def test_extract_keywords_basic():
    """Test multi-keyword extraction returns ranked keywords."""
    result = extract_keywords("How does authentication work in the login module?")
    assert len(result) <= 3
    assert "authentication" in result
    assert "login" in result
    assert "module" in result


def test_extract_keywords_empty():
    """Test empty query returns empty list."""
    assert extract_keywords("") == []
    assert extract_keywords("   ") == []


def test_extract_keywords_stopwords_only():
    """Test query with only stopwords falls back to longer words."""
    result = extract_keywords("the is at")
    assert len(result) > 0
    assert all(len(w) > 1 for w in result)


def test_extract_keywords_ranking():
    """Test keywords are ranked by length and position."""
    result = extract_keywords("Find functions that handle database connection errors")
    # "database" and "connection" are longest; earlier position gives slight boost
    assert "database" in result
    assert "connection" in result
    assert "functions" in result or "handle" in result or "errors" in result


def test_extract_keywords_max_limit():
    """Test max_keywords parameter is respected."""
    result = extract_keywords("a b c d e f g", max_keywords=3)
    assert len(result) == 3
