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
    collection_name="SyntheticBrightBiology",
    query_content_key="simulated_user_query",
    query_id_key="dataset_id",
    dataset_ids_key="dataset_id",
)

query_agent_benchmarking.run_eval(
    docs_collection=docs_collection,
    queries=queries_collection,
)