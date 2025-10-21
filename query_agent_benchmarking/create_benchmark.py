import os
from typing import Optional
import time

import weaviate
from weaviate.classes.config import DataType, Property, Configure
from weaviate.agents.classes import Operations
from weaviate.agents.transformation import TransformationAgent

def create_benchmark(
    docs_source_collection: str,
    benchmark_collection_name: str,
    delete_if_exists: Optional[bool] = True,
    num_queries: Optional[int] = 100,
    query_property_name: Optional[str] = "simulated_user_query",
    content_property_name: Optional[str] = "content",
    id_property_name: Optional[str] = "dataset_id",
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
    for i, obj in enumerate(source_collection.iterator()):
        if i >= num_queries:
            break
        obj_properties = obj.properties
        
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
        instruction="""You are tasked with generating reasoning-intensive queries for retrieval tasks. These queries should require intensive reasoning to identify that the provided document content is relevant—simple keyword matching or semantic similarity should NOT be sufficient.

    ## What Makes a Query "Reasoning-Intensive"?

    A reasoning-intensive query requires one or more of these reasoning types to connect it to the document:

    1. **Deductive Reasoning**: The document describes a general principle, theorem, or mechanism that can be applied to solve a specific problem or explain a specific scenario in the query.

    2. **Analogical Reasoning**: The document presents a parallel situation that uses similar underlying logic, algorithms, or approaches, even if the surface-level context appears different.

    3. **Causal Reasoning**: The document provides the cause or explanation for a problem/phenomenon described in the query, or vice versa.

    4. **Analytical Reasoning**: The document provides critical concepts, components, or knowledge that form essential pieces of a reasoning chain needed to solve the query problem.

    ## Requirements for Your Generated Query

    Given the document content, generate a query that:

    **MUST have LOW lexical overlap** - Avoid copying phrases or technical terms directly from the document
    **MUST have LOW semantic similarity** - Don't just paraphrase the document; create a genuinely different scenario
    **MUST require reasoning** - The connection should require understanding underlying principles, not surface matching
    **Should be realistic** - Frame it as a natural question someone would actually ask (troubleshooting, problem-solving, learning)
    **Should be specific and detailed** - Include context, constraints, or details that make it concrete
    **Should be standalone** - The query should be understandable without seeing the document

    ## Techniques to Ensure Reasoning-Intensive Connection

    - **Ground in different domains**: If the document is about plant biology, ask about tree maintenance
    - **Use different terminology**: If the document mentions "soluble salts," the query could discuss "dissolved minerals"
    - **Present specific scenarios**: Instead of asking "what is X?", describe a situation where X is the underlying cause
    - **Focus on application**: Ask how to solve a problem where the document's principle applies
    - **Mask the connection**: Make it non-obvious that the document's concept is what's needed

    Based on the provided document content, generate a single reasoning-intensive query that:
    1. Would realistically require this document to answer
    2. Has minimal keyword/semantic overlap with the document  
    3. Requires genuine reasoning to identify the document as relevant
    4. Is concrete, specific, and well-contextualized

    Output only the query text, nothing else."""
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