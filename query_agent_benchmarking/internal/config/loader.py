"""YAML config loading and merging utilities."""

import os
from typing import Any

import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    if config_path is None or not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config or {}


def merge_configs(file_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    """Merge file-based config with programmatic overrides.

    None values in override_config are ignored. If ``docs_collection`` is provided
    as an override, ``search_dataset`` is removed from the base config.
    """
    merged = file_config.copy()

    filtered_overrides = {k: v for k, v in override_config.items() if v is not None}

    if 'docs_collection' in filtered_overrides and 'search_dataset' in merged:
        del merged['search_dataset']

    merged.update(filtered_overrides)
    return merged
