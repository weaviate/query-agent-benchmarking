import query_agent_benchmarking

from query_agent_benchmarking.models import (
    DocsCollection,
    QueriesCollection,
    InMemoryQuery,
)

docs_collection = DocsCollection(
    name="BrightBiology",
    id_key="dataset_id",
)

queries_collection = QueriesCollection(
    name="BrightBiologyQueries",
    id_key="dataset_id",
)

query_agent_benchmarking.run_eval(
    dataset="bright/biology",
    docs_collection=docs_collection,
    queries_collection=queries_collection,
)