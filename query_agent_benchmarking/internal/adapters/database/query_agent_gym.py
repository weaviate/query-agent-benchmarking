"""Weaviate collection specs for QueryAgentGym, built from the dataset's own schemas.

Every other benchmark declares its collection schema in ``database_registry``.
QueryAgentGym ships one ``schema.json`` per environment instead, so the specs
here are derived from that file at load time: one ``DatasetSpec`` per
collection, with collection and property descriptions carried over so the
Query Agent sees them.
"""

from typing import Any, Callable, Mapping

import weaviate.collections.classes.config as wvcc

from .property_builder import DatasetSpecBuilder
from .spec import DatasetSpec
from .naming import add_tag_to_name
from query_agent_benchmarking.internal.adapters.dataset.query_agent_gym_loader import (
    load_query_agent_gym_schema,
    parse_environment,
)


DataType = wvcc.DataType

# schema.json data_type -> (Weaviate data type, is text)
_DATA_TYPES: dict[str, tuple[DataType, bool]] = {
    "searchable_text": (DataType.TEXT, True),
    "filterable_text": (DataType.TEXT, True),
    "int": (DataType.INT, False),
    "number": (DataType.NUMBER, False),
    "boolean": (DataType.BOOL, False),
    "date": (DataType.DATE, False),
    "text_array": (DataType.TEXT_ARRAY, True),
    "int_array": (DataType.INT_ARRAY, False),
    "number_array": (DataType.NUMBER_ARRAY, False),
    "boolean_array": (DataType.BOOL_ARRAY, False),
    "date_array": (DataType.DATE_ARRAY, False),
}


def _to_rfc3339(date: str) -> str:
    """``YYYY-MM-DD`` -> ``YYYY-MM-DDT00:00:00Z``, as Weaviate date properties require."""
    return date if "T" in date else f"{date}T00:00:00Z"


def _build_property(prop: dict[str, Any]) -> wvcc.Property:
    schema_type = prop["data_type"]
    if schema_type not in _DATA_TYPES:
        raise ValueError(f"Unsupported QueryAgentGym data_type: {schema_type!r}")
    data_type, is_text = _DATA_TYPES[schema_type]
    searchable = schema_type == "searchable_text"
    kwargs: dict[str, Any] = {}
    if is_text:
        # Identifiers, references and categories match on the whole value, so
        # `=` and LIKE behave like the ground truth's SQL semantics.
        kwargs["tokenization"] = wvcc.Tokenization.WORD if searchable else wvcc.Tokenization.FIELD
        kwargs["skip_vectorization"] = not searchable
    return wvcc.Property(
        name=prop["name"],
        data_type=data_type,
        description=prop.get("description"),
        **kwargs,
    )


def _build_mapper(properties: list[dict[str, Any]]) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    date_props = {p["name"] for p in properties if p["data_type"] == "date"}
    date_array_props = {p["name"] for p in properties if p["data_type"] == "date_array"}
    names = [p["name"] for p in properties]

    def mapper(item: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for name in names:
            value = item[name]
            if name in date_props:
                value = _to_rfc3339(value)
            elif name in date_array_props:
                value = [_to_rfc3339(v) for v in value]
            result[name] = value
        return result

    return mapper


def _build_collection_spec(dataset_name: str, collection: dict[str, Any]) -> DatasetSpec:
    properties = collection["properties"]
    builder = (
        DatasetSpecBuilder(dataset_name)
        .with_static_name(collection["name"])
        .with_description(collection["description"])
        .with_uuid_field("uuid")
        .with_custom_mapper(_build_mapper(properties))
    )
    for prop in properties:
        builder.with_property(_build_property(prop))

    searchable = [p["name"] for p in properties if p["data_type"] == "searchable_text"]
    if searchable:
        # Only the prose a search should match is embedded.
        builder.with_text2vec(name="default", source_properties=searchable)
    else:
        # Purely transactional collections have nothing to embed.
        builder.with_custom_vector_config(wvcc.Configure.Vectors.self_provided())
    return builder.build()


def build_query_agent_gym_specs(dataset_name: str) -> list[DatasetSpec]:
    """Build one ``DatasetSpec`` per collection of a QueryAgentGym environment."""
    schema = load_query_agent_gym_schema(parse_environment(dataset_name))
    return [_build_collection_spec(dataset_name, c) for c in schema["collections"]]


def query_agent_gym_collection_names(dataset_name: str, tag: str = "Default") -> dict[str, str]:
    """Map each tagged Weaviate collection name to its dataset collection name.

    E.g. ``{"SurfBeaches_Default": "SurfBeaches", ...}``.
    """
    schema = load_query_agent_gym_schema(parse_environment(dataset_name))
    return {add_tag_to_name(c["name"], tag): c["name"] for c in schema["collections"]}
