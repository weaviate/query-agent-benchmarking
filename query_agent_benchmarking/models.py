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
    name: str
    id_key: str

class QueriesCollection(BaseModel):
    name: str
    id_key: str