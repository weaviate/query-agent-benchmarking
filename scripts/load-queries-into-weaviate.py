# This is used to demonstrate running evals from Queries stored in Weaviate
import os
import weaviate
from weaviate.classes.config import Configure, Property

from query_agent_benchmarking.dataset import in_memory_dataset_loader

collection_name="BrightBiologyQueries"

_, queries = in_memory_dataset_loader("bright/biology")

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
)

if not weaviate_client.collections.exists(collection_name):
    weaviate_client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
        properties=[
            Property(
                name="question",
                data_type=weaviate.DataType.TEXT
            ),
            Property(
                name="query_id",
                data_type=weaviate.DataType.TEXT,
                skip_vectorization=True
            ),
            Property(
                name="dataset_ids",
                data_type=weaviate.DataType.TEXT_ARRAY,
                skip_vectorization=True
            )
        ]
    )

queries_collection = weaviate_client.collections.get(collection_name)

for query in queries:
    queries_collection.data.insert(
        properties={
            "question": query.question,
            "query_id": query.query_id,
            "dataset_ids": query.dataset_ids
        }
    )

print(f"Loaded {len(queries)} queries into Weaviate collection {collection_name}")