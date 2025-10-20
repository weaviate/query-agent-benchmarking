from pydantic import BaseModel

class ObjectID(BaseModel):
    object_id: str

class InMemoryQuery(BaseModel):
    question: str
    query_id: str
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
    query_id_key: str
    dataset_ids_key: str