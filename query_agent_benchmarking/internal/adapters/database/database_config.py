"""Supported datasets for the database loader.

This list enumerates all concrete dataset names that can be used as
``dataset_name`` in ``database_loader_config.yml``. Wildcard registry
patterns (e.g. ``beir/*``) are expanded into their known subsets here.
"""

supported_database_datasets = (
    "beir/fiqa/test",
    "beir/nq",
    "beir/scifact/test",
    "bright/biology",
    "bright/earth_science",
    "bright/economics",
    "bright/psychology",
    "bright/robotics",
    "enron",
    "filtered-cars",
    "freshstack-angular",
    "freshstack-godot",
    "freshstack-langchain",
    "freshstack-laravel",
    "freshstack-yolo",
    "irpapers",
    "irpapers-text-only",
    "longmemeval-m",
    "longmemeval-s",
    "lotte/lifestyle/test/forum",
    "lotte/lifestyle/test/search",
    "lotte/recreation/test/forum",
    "lotte/recreation/test/search",
    "multihoprag",
    "officeqa",
    "vidore_v3_hr",
    "wixqa",
    "browsecomp-plus",
)


def validate_database_dataset(dataset_name: str) -> None:
    """Raise ValueError if dataset_name is not a supported database dataset."""
    if dataset_name not in supported_database_datasets:
        raise ValueError(
            f"Unsupported dataset_name: '{dataset_name}'. "
            f"Supported datasets:\n  " + "\n  ".join(supported_database_datasets)
        )
