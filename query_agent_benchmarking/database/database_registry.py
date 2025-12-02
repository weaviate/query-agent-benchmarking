from .spec import DatasetSpec
from .property_builder import DatasetSpecBuilder

BenchmarkRegistry: list[DatasetSpec] = [
    (DatasetSpecBuilder("enron")
        .with_static_name("EnronEmails")
        .with_text_property("email_body")
        .with_dataset_id()
        .with_text2vec_weaviate()
        .build()),

    (DatasetSpecBuilder("beir/*")
        .with_prefix_name("Beir")
        .with_text_property("title")
        .with_text_property("content")
        .with_dataset_id(source_field="doc_id")
        .with_text2vec_weaviate()
        .build()),

    (DatasetSpecBuilder("bright/*")
        .with_prefix_name("Bright")
        .with_text_property("content")
        .with_dataset_id()
        .with_text2vec_weaviate()
        .build()),

    (DatasetSpecBuilder("lotte/*")
        .with_prefix_name("Lotte")
        .with_text_property("content", source_field="text")
        .with_dataset_id(source_field="doc_id")
        .with_text2vec_weaviate()
        .build()),

    (DatasetSpecBuilder("wixqa")
        .with_static_name("WixKB")
        .with_text_property("contents")
        .with_text_property("title")
        .with_text_property("article_type", searchable=False, filterable=False)
        .with_dataset_id(source_field="id")
        .with_text2vec_weaviate()
        .build()),

    (DatasetSpecBuilder("freshstack-*")
        .with_prefix_name("Freshstack", split_char="-")
        .with_text_property("docs_text", source_field="text")
        .with_dataset_id()
        .with_text2vec_weaviate()
        .build()),

    (DatasetSpecBuilder("irpapers/images")
        .with_static_name("IRPapersImages")
        .with_blob_property("base64_str")
        .with_dataset_id()
        .with_multi2vec_weaviate(
            image_field="base64_str",
        )
        .build()),

    (DatasetSpecBuilder("irpapers/text")
        .with_static_name("IRPapersText")
        .with_text_property("content", source_field="transcription")
        .with_dataset_id()
        .with_text2vec_weaviate()
        .build()),

    (DatasetSpecBuilder("multihoprag")
        .with_static_name("MultiHopRAG")
        .with_text_property("content", source_field="body")
        .with_dataset_id()
        .with_text2vec_weaviate()
        .build()),
]


def resolve_spec(dataset_name: str) -> DatasetSpec:
    """Find the spec that matches the given dataset name."""
    for spec in BenchmarkRegistry:
        if spec.matches(dataset_name):
            return spec
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")