from pydantic import BaseModel

class ObjectID(BaseModel):
    object_id: str

class InMemoryQuery(BaseModel):
    question: str
    dataset_ids: list[str]

class QueryResult(BaseModel):
    query: InMemoryQuery
    query_ground_truth_id: list[str]
    retrieved_ids: list[ObjectID]
    time_taken: float

class DocsCollection(BaseModel):
    collection_name: str
    content_key: str
    id_key: str

class QueriesCollection(BaseModel):
    collection_name: str
    query_content_key: str
    gold_ids_key: str

# WIP: Not sure this is the best way to store the hard negatives
# This tells the query-agent-benchmarking package how to store the hard negatives for each query
class HardNegativesCollection(BaseModel):
    collection_name: str
    query_content_key: str  # e.g., "query"
    gold_ids_key: str  # e.g., "gold_doc_ids" - stores list of gold IDs
    gold_documents_key: str  # e.g., "gold_documents" - stores list/text of gold doc contents
    hard_negative_document_key: str  # e.g., "hard_negative_doc" - stores the hard negative doc content
    hard_negative_id_key: str  # e.g., "hard_negative_id" - stores the hard negative doc ID