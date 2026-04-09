"""Dataset loaders that read from Weaviate collections."""

from typing import Optional

from query_agent_benchmarking.domain.models import InMemoryQuery, InMemoryAskQuery
from query_agent_benchmarking.adapters.clients.weaviate_client import get_weaviate_client


def load_queries_from_weaviate_collection(
    collection_name: str,
    query_content_key: str,
    gold_ids_key: str,
) -> list[InMemoryQuery]:
    """Load search queries from a custom Weaviate collection.

    Args:
        collection_name: Name of the Weaviate collection containing queries.
        query_content_key: Property name for the query text.
        gold_ids_key: Property name for the ground truth document IDs.

    Returns:
        List of InMemoryQuery instances.
    """
    weaviate_client = get_weaviate_client()

    query_collection = weaviate_client.collections.get(collection_name)

    queries: list[InMemoryQuery] = []
    for query_item in query_collection.iterator():
        props = query_item.properties
        queries.append(
            InMemoryQuery(
                question=props[query_content_key],
                dataset_ids=props[gold_ids_key],
            )
        )
    return queries


def load_ask_queries_from_weaviate(
    collection_name: str,
    query_content_key: str,
    answer_key: str,
    oracle_context_id_key: Optional[str] = None,
) -> list[InMemoryAskQuery]:
    """Load ask queries from a custom Weaviate collection.

    Args:
        collection_name: Name of the Weaviate collection.
        query_content_key: Property name for the question text.
        answer_key: Property name for the ground truth answer.
        oracle_context_id_key: Optional property name for oracle context IDs.

    Returns:
        List of InMemoryAskQuery instances.
    """
    client = get_weaviate_client()

    try:
        collection = client.collections.get(collection_name)

        return_props = [query_content_key, answer_key]
        if oracle_context_id_key:
            return_props.append(oracle_context_id_key)

        response = collection.query.fetch_objects(
            return_properties=return_props,
            limit=10000,
        )

        queries = []
        for obj in response.objects:
            oracle_id = None
            if oracle_context_id_key:
                oracle_id = str(obj.properties.get(oracle_context_id_key))

            queries.append(
                InMemoryAskQuery(
                    question=obj.properties[query_content_key],
                    ground_truth_answer=obj.properties[answer_key],
                    oracle_context_id=oracle_id,
                )
            )

        return queries
    finally:
        client.close()
