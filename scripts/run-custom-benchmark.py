import query_agent_benchmarking

from query_agent_benchmarking.models import (
    DocsCollection,
    QueriesCollection,
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
    docs_collection=docs_collection,
    queries=queries_collection,
)