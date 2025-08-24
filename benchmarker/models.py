from pydantic import BaseModel

class ObjectID(BaseModel):
    object_id: str

class QueryResult(BaseModel):
    query: dict
    query_id: list[str]
    retrieved_ids: list[ObjectID]
    time_taken: float