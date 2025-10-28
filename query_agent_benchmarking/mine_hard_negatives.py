import os
import weaviate

from query_agent_benchmarking.models import (
    DocsCollection,
    QueriesCollection,
    HardNegativesCollection,
)
def mine_hard_negatives(
    docs_collection: DocsCollection,
    queries_collection: QueriesCollection,
    hard_negatives_collection: HardNegativesCollection,
    hard_negative_key: str,
    negatives_per_query: int,
    query_samples: int = 100,
):
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )


    _queries_collection = weaviate_client.collections.get(queries_collection.collection_name)
    _docs_collection = weaviate_client.collections.get(docs_collection.collection_name)
    _hard_negatives_collection = weaviate_client.collections.get(hard_negatives_collection.collection_name)

    # populate the hard negatives collection with the hard negatives for each query
    for query in _queries_collection.iterator():
        query_properties = query.properties
        query_content = query_properties[queries_collection.query_content_key]
        gold_ids = query_properties[queries_collection.gold_ids_key]

        # update to ablate `retriever`
        results = _docs_collection.query.hybrid(
            query=query_content,
            limit=negatives_per_query*2,
        )

        for result in results:
            result_properties = result.properties
            result_id = result_properties[docs_collection.id_key]
            if result_id not in gold_ids:
                _hard_negatives_collection.data.insert(
                    properties={
                        hard_negatives_collection.query_content_key: query_content,
                        hard_negatives_collection.gold_ids_key: gold_ids,
                        hard_negatives_collection.gold_documents_key: [result_id],
                    }
                )

    # check if hard negatives are indeed not relevant with TA


    # delete hard negatives that are actually relevant as determined by TA