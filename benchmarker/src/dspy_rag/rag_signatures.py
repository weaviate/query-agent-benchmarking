from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import dspy

class Source(BaseModel):
    object_id: str

class AgentRAGResponse(BaseModel):
    final_answer: str
    sources: list[Source]
    searches: Optional[list[str]] = None
    aggregations: Optional[list] = None
    usage: Optional[Dict[str, Any]] = None

class DSPyAgentRAGResponse(dspy.Prediction):
    """
    DSPy-compatible version of AgentRAGResponse that inherits from dspy.Prediction.
    
    This class provides the set_lm_usage method that DSPy expects while maintaining
    compatibility with the existing benchmark infrastructure that expects AgentRAGResponse interface.
    """
    
    def __init__(self, final_answer: str = "", sources: List[Source] = None, 
                 searches: Optional[List[str]] = None, aggregations: Optional[List] = None,
                 usage: Optional[Dict[str, Any]] = None, **kwargs):
        # Initialize the parent dspy.Prediction class
        super().__init__(**kwargs)
        
        # Set our custom attributes
        self.final_answer = final_answer
        self.sources = sources or []
        self.searches = searches
        self.aggregations = aggregations
        self.usage = usage or {}
    
    def to_agent_rag_response(self) -> AgentRAGResponse:
        """Convert to the original AgentRAGResponse for compatibility with benchmark infrastructure."""
        return AgentRAGResponse(
            final_answer=self.final_answer,
            sources=self.sources,
            searches=self.searches,
            aggregations=self.aggregations,
            usage=self.usage
        )
    
    @classmethod
    def from_agent_rag_response(cls, response: AgentRAGResponse) -> 'DSPyAgentRAGResponse':
        """Create DSPyAgentRAGResponse from AgentRAGResponse."""
        return cls(
            final_answer=response.final_answer,
            sources=response.sources,
            searches=response.searches,
            aggregations=response.aggregations,
            usage=response.usage
        )
    
    # Provide dict-like access for compatibility
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)

class GenerateAnswer(dspy.Signature):
    """Assess the context and answer the question."""

    question: str = dspy.InputField()
    contexts: str = dspy.InputField()
    final_answer: str = dspy.OutputField()

class WriteSearchQueries(dspy.Signature):
    """Write search queries to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    search_queries: list[str] = dspy.OutputField()

# TODO: Maybe extend to enable multiple filters with one search query
class SearchQueryWithFilter(BaseModel):
    search_query: str
    filter: Optional[str]

class WriteSearchQueriesWithFilters(dspy.Signature):
    """Write search queries with optional filters to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    filters_available: str = dspy.InputField()
    search_queries_with_filters: SearchQueryWithFilter = dspy.OutputField()

class FilterIrrelevantSearchResults(dspy.Signature):
    """Filter out search results that are not relevant to answering the question."""
    
    question: str = dspy.InputField()
    search_results: dict[int, str] = dspy.InputField(desc="The search results keyed by their id.")
    filtered_results: list[int] = dspy.OutputField(desc="The ids of relevant results.")

class SummarizeSearchResults(dspy.Signature):
    """Summarize search results to extract the most important information related to the question."""
    
    question: str = dspy.InputField()
    search_results: dict[int, str] = dspy.InputField()
    summary: str = dspy.OutputField() # add citations to the ids in the summary
