"""Database schema and loading utilities for Weaviate collections."""
from .spec import DatasetSpec
from .property_builder import DatasetSpecBuilder
from .database_registry import BenchmarkRegistry, resolve_spec
from .database_loader import (
    database_loader,
    create_collection_with_vector_config,
    get_vector_config,
)
from .database_config import supported_database_datasets, validate_database_dataset
from .engram_loader import (
    engram_ingest_tenant,
    engram_ingest_all_tenants,
    TenantIngestionStats,
)

__all__ = [
    "DatasetSpec",
    "DatasetSpecBuilder",
    "BenchmarkRegistry",
    "resolve_spec",
    "database_loader",
    "create_collection_with_vector_config",
    "get_vector_config",
    "supported_database_datasets",
    "validate_database_dataset",
    "engram_ingest_tenant",
    "engram_ingest_all_tenants",
    "TenantIngestionStats",
]