from pydantic import BaseModel
from typing import Literal, Optional

class IntPropertyFilter(BaseModel):
    property_name: str
    operator: Literal["=", "<", ">", "<=", ">="]
    value: int | float

class TextPropertyFilter(BaseModel):
    property_name: str
    operator: Literal["=", "LIKE"]
    value: str

class BooleanPropertyFilter(BaseModel):
    property_name: str
    operator: Literal["=", "!="]
    value: bool

class IntAggregation(BaseModel):
    property_name: str
    metrics: Literal["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]

class TextAggregation(BaseModel):
    property_name: str
    metrics: Literal["TOP_OCCURRENCES"]
    top_occurrences_limit: Optional[int] = None

class BooleanAggregation(BaseModel):
    property_name: str
    metrics: Literal["TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]

class DateAggregation(BaseModel):
    property_name: str
    metrics: Literal["MIN", "MAX", "MEAN", "MEDIAN", "MODE"]

class WeaviateQuery(BaseModel):
    corresponding_natural_language_query: Optional[str] = None
    target_collection: str
    search_query: Optional[str] = None
    limit: Optional[int] = 5
    integer_property_filter: Optional[IntPropertyFilter] = None
    text_property_filter: Optional[TextPropertyFilter] = None
    boolean_property_filter: Optional[BooleanPropertyFilter] = None
    integer_property_aggregation: Optional[IntAggregation] = None
    text_property_aggregation: Optional[TextAggregation] = None
    boolean_property_aggregation: Optional[BooleanAggregation] = None
    groupby_property: Optional[str] = None
    total_count: Optional[bool] = None