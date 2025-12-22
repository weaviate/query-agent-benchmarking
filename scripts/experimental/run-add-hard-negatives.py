import query_agent_benchmarking

from query_agent_benchmarking.models import (
    DocsCollection,
    QueriesCollection,
    HardNegativesCollection,
)

docs_collection = DocsCollection(
    collection_name="BrightBiology", # name of the collection to search through
    content_key="content", # searchable property name
    id_key="dataset_id", # dataset id property
)

queries_collection = QueriesCollection(
    collection_name="SyntheticBrightBiology", # name of collection containing queries
    query_content_key="simulated_user_query", # property name containing the query
    gold_ids_key="dataset_ids", # property name containing the document ids the query should return
)

hard_negatives_collection = HardNegativesCollection(
    collection_name="HardNegatives", # name of collection containing hard negatives
    query_content_key="query", # property name containing the query
    gold_ids_key="gold_doc_ids", # property name containing the document ids the query should return
    gold_documents_key="gold_documents", # property name containing the document contents
    hard_negative_document_key="hard_negative_doc", # property name containing the hard negative document content
    hard_negative_id_key="hard_negative_id", # property name containing the hard negative document id
)

query_agent_benchmarking.add_hard_negatives(
    negatives_per_query=5,
    docs_collection=docs_collection,
    queries_collection=queries_collection,
    hard_negatives_collection=hard_negatives_collection,
    query_samples=100,
)