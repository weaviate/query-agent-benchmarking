"""Port interfaces for dataset loading."""

from typing import Protocol, runtime_checkable

from query_agent_benchmarking.internal.core.domain.models import InMemoryQuery, InMemoryAskQuery


@runtime_checkable
class SearchDatasetRepository(Protocol):
    """Protocol for loading search benchmark datasets."""

    def load_queries(self, dataset_name: str, **kwargs) -> list[InMemoryQuery]:
        """Load search queries with ground truth for a dataset.

        Args:
            dataset_name: Name of the dataset to load.
            **kwargs: Dataset-specific loading options.

        Returns:
            List of search queries with ground truth document IDs.
        """
        ...

    def load_corpus(self, dataset_name: str, **kwargs) -> list[dict]:
        """Load the document corpus for a dataset.

        Args:
            dataset_name: Name of the dataset to load.
            **kwargs: Dataset-specific loading options.

        Returns:
            List of document dictionaries.
        """
        ...


@runtime_checkable
class AskDatasetRepository(Protocol):
    """Protocol for loading ask benchmark datasets."""

    def load_queries(self, dataset_name: str, **kwargs) -> list[InMemoryAskQuery]:
        """Load ask queries with ground truth answers.

        Args:
            dataset_name: Name of the dataset to load.
            **kwargs: Dataset-specific loading options.

        Returns:
            List of ask queries with ground truth answers.
        """
        ...
