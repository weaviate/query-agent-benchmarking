import os
from typing import Optional
import time

import weaviate
from weaviate.classes.config import DataType, Property, Configure
from weaviate.agents.classes import Operations
from weaviate.agents.transformation import TransformationAgent

from query_agent_benchmarking.experimental.query_gen_prompts import bright_query_gen

def create_benchmark(
    docs_source_collection: str,
    benchmark_collection_name: str,
    delete_if_exists: Optional[bool] = True,
    num_queries: Optional[int] = 100,
    query_property_name: Optional[str] = "simulated_user_query",
    content_property_name: Optional[str] = "content",
    id_property_name: Optional[str] = "dataset_id",
    prune_short_docs: Optional[bool] = True,
):
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    if delete_if_exists and weaviate_client.collections.exists(benchmark_collection_name):
        print(f"\033[96mDeleting existing collection {benchmark_collection_name}...\033[0m")
        weaviate_client.collections.delete(benchmark_collection_name)

    if not weaviate_client.collections.exists(benchmark_collection_name):
      _create_collection(
        weaviate_client=weaviate_client, 
        name=benchmark_collection_name,
        query_property_name=query_property_name, 
        content_property_name=content_property_name, 
        id_property_name=id_property_name
    )

    benchmark_collection = weaviate_client.collections.get(benchmark_collection_name)

    source_collection = weaviate_client.collections.get(docs_source_collection)
    obj_counter = 0
    for obj in source_collection.iterator():
        if obj_counter >= num_queries:
            break
        if prune_short_docs and len(obj.properties[content_property_name].split(" ")) < 100:
            continue
        obj_properties = obj.properties
        obj_counter += 1
        properties={
            content_property_name: obj_properties[content_property_name],
            "dataset_ids": [str(obj_properties[id_property_name])],
        }

        benchmark_collection.data.insert(
            properties=properties
        )

    create_reasoning_intensive_queries = Operations.update_property(
        property_name=query_property_name,
        view_properties=[content_property_name],
        instruction=bright_query_gen,
    )

    agent = TransformationAgent(
        client=weaviate_client,
        collection=benchmark_collection_name,
        operations=[create_reasoning_intensive_queries],
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

    print(f"\033[92mWorkflow {workflow_id} finished.\033[0m")
    print(agent.get_status(workflow_id))

def _create_collection(
    weaviate_client: weaviate.WeaviateClient, 
    name: str, 
    query_property_name: str, 
    content_property_name: str, 
    id_property_name: str
):
    """Helper to create collection with standard schema."""
    return weaviate_client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
        properties=[
            Property(
                name=query_property_name,
                data_type=DataType.TEXT,
            ),
            Property(
                name=content_property_name,
                data_type=DataType.TEXT,
                skip_vectorization=True,
            ),
            Property(
                name="dataset_ids",
                data_type=DataType.TEXT_ARRAY,
                skip_vectorization=True,
            ),
        ]
    )