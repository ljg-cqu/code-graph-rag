"""Class utility functions."""


def is_same_class(obj1: object, obj2: object) -> bool:
    """Check if two objects are instances of the same class.

    Args:
        obj1: First object to compare.
        obj2: Second object to compare.

    Returns:
        True if both objects are instances of the same class, False otherwise.

    Example:
        >>> is_same_class(1, 2)
        True
        >>> is_same_class(1, "hello")
        False
        >>> is_same_class([1, 2], [3, 4])
        True
    """
    return type(obj1) is type(obj2)
