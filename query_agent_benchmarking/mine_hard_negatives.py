import os
import weaviate

from query_agent_benchmarking.models import (
    DocsCollection,
    QueriesCollection,
)
def mine_hard_negatives(
    docs_collection: DocsCollection,
    queries_collection: QueriesCollection,
    hard_negatives_collection: HardNegativesCollection,
    hard_negative_key: str,
    negatives_per_query: int,
):
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
