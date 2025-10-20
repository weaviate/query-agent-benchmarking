import query_agent_benchmarking

from query_agent_benchmarking.models import (
    DocsCollection,
    QueriesCollection,
)

docs_collection = DocsCollection(
    collection_name="BrightBiology",
    content_key="content",
    id_key="dataset_id",
)

queries_collection = QueriesCollection(
    collection_name="BrightBiologyQueries",
    query_content_key="question",
    query_id_key="query_id",
    dataset_ids_key="dataset_ids",
)

query_agent_benchmarking.run_eval(
    docs_collection=docs_collection,
    queries=queries_collection,
)