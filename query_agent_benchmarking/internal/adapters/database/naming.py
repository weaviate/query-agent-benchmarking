"""Collection naming utilities for database adapters."""

import re


def pascalize_name(raw: str) -> str:
    """Convert a raw string to PascalCase by splitting on non-alphanumeric chars.

    Example: "beir/scifact" -> "BeirScifact"
    """
    tokens = re.split(r"[^0-9A-Za-z]+", raw)
    return "".join(t.capitalize() for t in tokens if t)


def add_tag_to_name(original_name: str, tag: str) -> str:
    """Append a tag suffix to a collection name."""
    return f"{original_name}_{tag}"
