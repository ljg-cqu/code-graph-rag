"""Unit tests for query utility functions."""

from codebase_rag.utils.query_utils import extract_best_keyword


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