import os
import time

import weaviate
from weaviate.classes.config import DataType, Property, Configure
from weaviate.classes.query import Filter
from weaviate.agents.classes import Operations
from weaviate.agents.transformation import TransformationAgent

from query_agent_benchmarking.models import (
    DocsCollection,
    QueriesCollection,
    HardNegativesCollection,
)
from query_agent_benchmarking.hard_negative_prompts import relevance_assessment_prompt

def add_hard_negatives(
    docs_collection: DocsCollection,
    queries_collection: QueriesCollection,
    hard_negatives_collection: HardNegativesCollection,
    negatives_per_query: int,
    query_samples: int = 100,
):
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )


    _queries_collection = weaviate_client.collections.get(queries_collection.collection_name)
    _docs_collection = weaviate_client.collections.get(docs_collection.collection_name)

    if not weaviate_client.collections.exists(hard_negatives_collection.collection_name):
        _hard_negatives_collection = _create_hard_negatives_collection(
            weaviate_client=weaviate_client,
            name=hard_negatives_collection.collection_name,
            query_content_key=hard_negatives_collection.query_content_key,
            gold_ids_key=hard_negatives_collection.gold_ids_key,
            gold_documents_key=hard_negatives_collection.gold_documents_key,
            hard_negative_document_key=hard_negatives_collection.hard_negative_document_key,
            hard_negative_id_key=hard_negatives_collection.hard_negative_id_key,
        )
    else:
        _hard_negatives_collection = weaviate_client.collections.get(hard_negatives_collection.collection_name)

    # populate the hard negatives collection with the hard negatives for each query
    query_counter = 0
    for query in _queries_collection.iterator():
        if query_counter >= query_samples:
            break
        query_counter += 1
        query_properties = query.properties
        query_content = query_properties[queries_collection.query_content_key]
        gold_ids = query_properties[queries_collection.gold_ids_key]

        # Fetch gold documents once per query
        gold_docs = []
        for gold_id in gold_ids:
            gold_response = _docs_collection.query.fetch_objects(
                filters=Filter.by_property(docs_collection.id_key).equal(gold_id),
                limit=1
            )
            if gold_response.objects:
                gold_doc_content = gold_response.objects[0].properties[docs_collection.content_key]
                gold_docs.append(gold_doc_content)

        # update to ablate `retriever`
        hard_negative_results = _docs_collection.query.hybrid(
            query=query_content,
            limit=negatives_per_query * 2,
        )
        collected_count = 0
        for result in hard_negative_results.objects:
            if collected_count >= negatives_per_query:
                break

            result_properties = result.properties
            result_id = result_properties[docs_collection.id_key]

            if result_id not in gold_ids:
                _hard_negatives_collection.data.insert(
                    properties={
                        hard_negatives_collection.query_content_key: query_content,
                        hard_negatives_collection.gold_ids_key: gold_ids,
                        hard_negatives_collection.gold_documents_key: gold_docs,  # reuse fetched gold docs
                        hard_negatives_collection.hard_negative_document_key: result_properties[docs_collection.content_key],
                        hard_negatives_collection.hard_negative_id_key: result_id,
                    }
                )
                collected_count += 1

    # check if hard negatives are indeed not relevant with TA
    assess_if_relevant = Operations.append_property(
        property_name="is_relevant",
        data_type=DataType.BOOL,
        view_properties=[
            hard_negatives_collection.query_content_key,
            hard_negatives_collection.hard_negative_document_key,
        ],
        instruction=relevance_assessment_prompt,
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
    
    print(f"\033[92mHard negatives added and assessed for relevance for {query_samples} queries.\033[0m")

def _create_hard_negatives_collection(
    weaviate_client: weaviate.WeaviateClient,
    name: str,
    query_content_key: str,
    gold_ids_key: str,
    gold_documents_key: str,
    hard_negative_document_key: str,
    hard_negative_id_key: str,
):
    _hard_negatives_collection = weaviate_client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
        properties=[
            Property(
                name=query_content_key,
                data_type=DataType.TEXT,
            ),
            Property(
                name=gold_ids_key,
                data_type=DataType.TEXT_ARRAY,
                skip_vectorization=True,
            ),
            Property(
                name=gold_documents_key,
                data_type=DataType.TEXT_ARRAY,
                skip_vectorization=True,
            ),
            Property(
                name=hard_negative_document_key,
                data_type=DataType.TEXT,
                skip_vectorization=True,
            ),
            Property(
                name=hard_negative_id_key,
                data_type=DataType.TEXT,
                skip_vectorization=True,
            ),
         ],
    )
    return _hard_negatives_collection