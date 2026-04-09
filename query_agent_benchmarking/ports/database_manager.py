"""Port interface for database collection management."""

from typing import Any, Callable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class DatabaseManager(Protocol):
    """Protocol for managing database collections."""

    def create_collection(
        self,
        name: str,
        properties: Any,
        vector_config: Any,
        recreate: bool = True,
    ) -> None:
        """Create a collection with the given schema.

        Args:
            name: Collection name.
            properties: List of property configurations.
            vector_config: Vectorizer configuration.
            recreate: Whether to drop and recreate if exists.
        """
        ...

    def batch_insert(
        self,
        collection: str,
        items: Sequence[dict],
        item_to_props: Callable[[dict], dict],
    ) -> int:
        """Batch insert items into a collection.

        Args:
            collection: Target collection name.
            items: Sequence of items to insert.
            item_to_props: Function to transform items to collection properties.

        Returns:
            Number of items successfully inserted.
        """
        ...

    def close(self) -> None:
        """Close the database connection."""
        ...
