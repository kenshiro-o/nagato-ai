def starts_with(value, prefix):
    """Check if a string value starts with the given prefix."""
    if isinstance(value, str):
        return value.startswith(prefix)
    return False


def contains(value, substring):
    """Check if a string value contains the given substring."""
    if isinstance(value, str):
        return substring in value
    return False


def is_even(value, _):
    """Check if a number is even."""
    if isinstance(value, (int, float)):
        return value % 2 == 0
    return False


def is_odd(value, _):
    """Check if a number is odd."""
    if isinstance(value, (int, float)):
        return value % 2 != 0
    return False


def is_positive(value, _):
    """Check if a number is positive."""
    if isinstance(value, (int, float)):
        return value > 0
    return False


def is_negative(value, _):
    """Check if a number is negative."""
    if isinstance(value, (int, float)):
        return value < 0
    return False


def is_zero(value, _):
    """Check if a number is zero."""
    if isinstance(value, (int, float)):
        return value == 0
