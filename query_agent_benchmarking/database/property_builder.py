"""Fluent builder for DatasetSpec."""
from typing import Any, Callable, Dict, Mapping, Optional

import weaviate.collections.classes.config as wvcc

from .spec import DatasetSpec

# Property type constants
TEXT = wvcc.DataType.TEXT
BLOB = wvcc.DataType.BLOB
INT = wvcc.DataType.INT
FIELD = wvcc.Tokenization.FIELD

def pascalize_name(name: str) -> str:
    """Convert a name to PascalCase."""
    return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))


def dataset_id_property() -> wvcc.Property:
    """Standard dataset_id property for filtering."""
    return wvcc.Property(
        name="dataset_id",
        data_type=TEXT,
        index_searchable=False,
        index_filterable=True,
        skip_vectorization=True,
        tokenization=FIELD,
    )


class DatasetSpecBuilder:
    """Fluent builder for DatasetSpec."""

    def __init__(self, pattern: str):
        self._pattern = pattern
        self._name_fn: Optional[Callable[[str], str]] = None
        self._properties: list[wvcc.Property] = []
        self._vector_config: Any = None
        self._field_mappings: dict[str, str] = {}
        self._id_field: str = "dataset_id"
        self._custom_mapper: Optional[Callable[[Mapping[str, Any]], Dict[str, Any]]] = None

    # --- Name configuration ---

    def with_name(self, name_fn: Callable[[str], str]) -> "DatasetSpecBuilder":
        """Set a custom collection name function."""
        self._name_fn = name_fn
        return self

    def with_static_name(self, name: str) -> "DatasetSpecBuilder":
        """Set a static collection name."""
        self._name_fn = lambda _: name
        return self

    def with_prefix_name(
        self,
        prefix: str,
        split_char: str = "/",
        split_index: int = 1,
    ) -> "DatasetSpecBuilder":
        """
        Set a name derived from the dataset name.
        
        E.g., "beir/scifact" with prefix="Beir" -> "BeirScifact"
        """
        self._name_fn = lambda d: f"{prefix}{pascalize_name(d.split(split_char)[split_index])}"
        return self

    # --- Properties ---

    def with_text_property(
        self,
        name: str,
        source_field: Optional[str] = None,
        searchable: bool = True,
        filterable: bool = True,
    ) -> "DatasetSpecBuilder":
        """Add a text property."""
        self._properties.append(
            wvcc.Property(
                name=name,
                data_type=TEXT,
                index_searchable=searchable,
                index_filterable=filterable,
            )
        )
        self._field_mappings[name] = source_field or name
        return self

    def with_blob_property(
        self,
        name: str,
        source_field: Optional[str] = None,
    ) -> "DatasetSpecBuilder":
        """Add a blob property for binary data."""
        self._properties.append(wvcc.Property(name=name, data_type=BLOB))
        self._field_mappings[name] = source_field or name
        return self

    def with_int_property(
        self,
        name: str,
        source_field: Optional[str] = None,
        filterable: bool = True,
    ) -> "DatasetSpecBuilder":
        """Add an integer property."""
        self._properties.append(
            wvcc.Property(name=name, data_type=INT, index_filterable=filterable)
        )
        self._field_mappings[name] = source_field or name
        return self

    def with_dataset_id(self, source_field: str = "dataset_id") -> "DatasetSpecBuilder":
        """Add the standard dataset_id property for filtering."""
        self._properties.append(dataset_id_property())
        self._id_field = source_field
        return self

    # --- Vectorizers ---

    def with_text2vec_weaviate(self, model: Optional[str] = None) -> "DatasetSpecBuilder":
        """Use text2vec_weaviate vectorizer."""
        if model:
            self._vector_config = wvcc.Configure.Vectorizer.text2vec_weaviate(model=model)
        else:
            self._vector_config = wvcc.Configure.Vectorizer.text2vec_weaviate()
        return self

    def with_multi2vec_weaviate(
        self,
        image_field: str,
        model: str,
        dynamic_ef_factor: int = 32,
    ) -> "DatasetSpecBuilder":
        """Use multi2vec_weaviate vectorizer for images."""
        self._vector_config = wvcc.Configure.MultiVectors.multi2vec_weaviate(
            image_field=image_field,
            model=model,
            encoding=wvcc.Configure.VectorIndex.MultiVector.Encoding.muvera(),
            vector_index_config=wvcc.Configure.VectorIndex.hnsw(
                dynamic_ef_factor=dynamic_ef_factor,
            ),
        )
        return self

    def with_custom_vector_config(self, config: Any) -> "DatasetSpecBuilder":
        """Use a custom vector configuration."""
        self._vector_config = config
        return self

    # --- Custom mapping ---

    def with_custom_mapper(
        self,
        mapper: Callable[[Mapping[str, Any]], Dict[str, Any]],
    ) -> "DatasetSpecBuilder":
        """Override the auto-generated property mapper."""
        self._custom_mapper = mapper
        return self

    # --- Build ---

    def build(self) -> DatasetSpec:
        """Build the final DatasetSpec."""
        if self._name_fn is None:
            raise ValueError("Name function is required")
        if self._vector_config is None:
            raise ValueError("Vector config is required")
        if not self._properties:
            raise ValueError("At least one property is required")

        item_to_props = self._custom_mapper or self._build_mapper()

        return DatasetSpec(
            pattern=self._pattern,
            name_fn=self._name_fn,
            properties=tuple(self._properties),
            vector_config=self._vector_config,
            item_to_props=item_to_props,
        )

    def _build_mapper(self) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
        """Build the property mapper from field mappings."""
        field_mappings = self._field_mappings.copy()
        id_field = self._id_field

        def mapper(item: Mapping[str, Any]) -> Dict[str, Any]:
            result = {}
            for prop_name, source_field in field_mappings.items():
                if source_field in item:
                    result[prop_name] = item[source_field]
            if id_field in item:
                result["dataset_id"] = str(item[id_field])
            return result

        return mapper