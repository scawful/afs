"""Hello-world AFS extension example."""

__all__ = ["greeting"]


def greeting() -> str:
    """Return a deterministic extension greeting."""
    return "hello from an AFS extension"
