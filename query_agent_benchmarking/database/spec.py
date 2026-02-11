"""Dataset specification definition."""
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import weaviate.collections.classes.config as wvcc


@dataclass
class DatasetSpec:
    """Defines how to store a dataset in Weaviate."""
    pattern: str
    name_fn: Callable[[str], str]
    properties: tuple[wvcc.Property, ...]
    vector_config: Any
    item_to_props: Callable[[Mapping[str, Any]], dict[str, Any]]

    def matches(self, dataset_name: str) -> bool:
        """Check if this spec matches a dataset name."""
        if self.pattern.endswith("/*"):
            prefix = self.pattern[:-2]
            return dataset_name.startswith(prefix + "/")
        elif self.pattern.endswith("-*"):
            prefix = self.pattern[:-2]
            return dataset_name.startswith(prefix + "-")
        return self.pattern == dataset_name