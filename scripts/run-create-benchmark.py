import query_agent_benchmarking

query_agent_benchmarking.create_benchmark(
    docs_source_collection="BrightBiology",
    benchmark_collection_name="SyntheticBrightBiology",
)

# Then optionally test with:
'''
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
    gold_ids_key="dataset_ids",
)

query_agent_benchmarking.run_eval(
    docs_collection=docs_collection,
    queries=queries_collection,
)
'''