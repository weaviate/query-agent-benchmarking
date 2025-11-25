"""Database schema and loading utilities for Weaviate collections."""
from .spec import DatasetSpec
from .property_builder import DatasetSpecBuilder
from .database_registry import REGISTRY, resolve_spec
from .database_loader import (
    database_loader,
    create_collection_with_vector_config,
    get_vector_config,
)

__all__ = [
    "DatasetSpec",
    "DatasetSpecBuilder",
    "REGISTRY",
    "resolve_spec",
    "database_loader",
    "create_collection_with_vector_config",
    "get_vector_config",
]