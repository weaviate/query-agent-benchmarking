import os
import time

import weaviate
from weaviate.classes.config import DataType
from weaviate.classes.query import Filter
from weaviate.agents.classes import Operations
from weaviate.agents.transformation import TransformationAgent

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
                # THIS ISN'T ALL THE DATA FOR THE HARD NEGATIVE!!
                _hard_negatives_collection.data.insert(
                    properties={
                        hard_negatives_collection.query_content_key: query_content,
                        hard_negatives_collection.gold_ids_key: gold_ids,
                        hard_negatives_collection.gold_documents_key: [result_id],
                    }
                )

    # check if hard negatives are indeed not relevant with TA
    assess_if_relevant = Operations.append_property(
        property_name="is_relevant",
        data_type=DataType.BOOL,
        view_properties=[hard_negatives_collection.gold_documents_key],
        instruction=f"Assess if the document is relevant to the query: {query_content}.",
    )

    agent = TransformationAgent(
        client=weaviate_client,
        collection=hard_negatives_collection.collection_name,
        operations=[assess_if_relevant],
    )
    response = agent.update_all()
    workflow_id = response.workflow_id

    finished = False

    while not finished:
        status = agent.get_status(workflow_id)
        if status["status"]["state"] != "running":
            finished = True
        else:
            print("\033[96mNot yet finished. Checking again in 30 seconds...\033[0m")
            time.sleep(30)

    # delete hard negatives that are actually relevant as determined by TA
    result = _hard_negatives_collection.data.delete_many(
        where=Filter.by_property(property_name="is_relevant", value=True),
    )

    print(result)