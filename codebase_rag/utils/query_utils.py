"""Query utility functions for keyword extraction and processing."""

STOPWORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
    'in', 'with', 'to', 'for', 'of', 'what', 'how', 'where', 'when',
    'why', 'who', 'show', 'find', 'tell', 'get', 'list', 'all', 'any',
    'some', 'does', 'do', 'can', 'could', 'would', 'should', 'will',
}


def extract_best_keyword(query: str) -> str:
    """Extract the most meaningful keyword from a natural language query.

    Filters stopwords and returns the longest remaining word.
    Falls back to longest word if no meaningful words found.
    Returns empty string if query is empty.

    Args:
        query: Natural language query string

    Returns:
        Best keyword extracted from query
    """
    if not query:
        return ""

    words = query.lower().split()
    meaningful = [w for w in words if len(w) > 2 and w not in STOPWORDS]

    if not meaningful:
        return max(words, key=len, default="")

    return max(meaningful, key=len)
