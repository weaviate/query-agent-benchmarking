from .spec import DatasetSpec
from .property_builder import DatasetSpecBuilder

BenchmarkRegistry: list[DatasetSpec] = [
    (DatasetSpecBuilder("enron")
        .with_static_name("EnronEmails")
        .with_text_property("email_body")
        .with_dataset_id()
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("beir/*")
        .with_prefix_name("Beir")
        .with_text_property("title")
        .with_text_property("content")
        .with_dataset_id(source_field="doc_id")
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("bright/*")
        .with_prefix_name("Bright")
        .with_text_property("content")
        .with_dataset_id()
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("lotte/*")
        .with_prefix_name("Lotte")
        .with_text_property("content", source_field="text")
        .with_dataset_id(source_field="doc_id")
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("wixqa")
        .with_static_name("WixKB")
        .with_text_property("contents")
        .with_text_property("title")
        .with_text_property("article_type", searchable=False, filterable=False)
        .with_dataset_id(source_field="id")
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("freshstack-*")
        .with_prefix_name("Freshstack", split_char="-")
        .with_text_property("docs_text", source_field="text")
        .with_dataset_id()
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("irpapers")
        .with_static_name("IRPapers")
        .with_text_property("content", source_field="transcription")
        .with_blob_property("image", source_field="base64_str")
        .with_dataset_id()
        .with_text2vec_provider_named(source_properties=["content"], base_name="text_content")
        .with_multi2vec(image_field="image", name="image_content", name_by_provider=True)
        .build()),

    (DatasetSpecBuilder("irpapers-text-only")
        .with_static_name("IRPapersTextOnly")
        .with_text_property("content", source_field="transcription")
        .with_dataset_id()
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("filtered-cars")
        .with_static_name("FilteredCars")
        .with_text_property("make")
        .with_text_property("model")
        .with_int_property("year")
        .with_text_property("fuel_type")
        .with_text_property("body_type")
        .with_int_property("mileage")
        .with_int_property("price")
        .with_text_property("color")
        .with_text_property("color_family")
        .with_text_property("transmission")
        .with_text_property("drivetrain")
        .with_dataset_id()
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("multihoprag")
        .with_static_name("MultiHopRAG")
        .with_text_property("content", source_field="body")
        .with_dataset_id()
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("vidore_v3_hr")
        .with_static_name("Vidore_v3_hr")
        .with_text_property("content", source_field="markdown")
        .with_blob_property("image", source_field="image_base64")
        .with_dataset_id()
        .with_text2vec_provider_named(source_properties=["content"], base_name="text_content")
        .with_multi2vec(image_field="image", name="image_content", name_by_provider=True)
        .build()),

    (DatasetSpecBuilder("longmemeval-s")
        .with_static_name("LongmemevalS")
        .with_text_property("session_text")
        .with_text_property("session_date", searchable=False, filterable=True)
        .with_dataset_id(source_field="session_id")
        .with_text2vec()
        .with_multi_tenancy(tenant_id_field="tenant_id")
        .build()),

    (DatasetSpecBuilder("longmemeval-m")
        .with_static_name("LongmemevalM")
        .with_text_property("session_text")
        .with_text_property("session_date", searchable=False, filterable=True)
        .with_dataset_id(source_field="session_id")
        .with_text2vec()
        .with_multi_tenancy(tenant_id_field="tenant_id")
        .build()),

    (DatasetSpecBuilder("officeqa")
        .with_static_name("OfficeQA")
        .with_blob_property("image", source_field="image")
        .with_text_property("source_pdf", searchable=False, filterable=True)
        .with_int_property("page_number", filterable=True)
        .with_multi2vec(image_field="image", embedding_model="cohere/embed-v4.0")
        .build()),

    (DatasetSpecBuilder("browsecomp-plus")
        .with_static_name("BrowseCompPlus")
        .with_text_property("content", source_field="text")
        .with_dataset_id(source_field="docid")
        .with_text2vec()
        .build()),

    (DatasetSpecBuilder("obliq-bench-*")
        .with_prefix_name("ObliqBench", split_char="-", split_index=2)
        .with_text_property("content", source_field="text")
        .with_dataset_id()
        .with_text2vec()
        .build()),

]


def resolve_spec(dataset_name: str) -> DatasetSpec:
    """Find the spec that matches the given dataset name."""
    for spec in BenchmarkRegistry:
        if spec.matches(dataset_name):
            return spec
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")
