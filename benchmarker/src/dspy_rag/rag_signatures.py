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

class RerankResults(dspy.Signature):
    """Determine a ranking of the passages based on how relevant they are to the query. 
If the query is a question, how relevant a passage is depends on how well it answers the question. 
If not, try analyze the intent of the query and assess how well each passage satisfy the intent. 
The query may have typos and passages may contain contradicting information. 
However, we do not get into fact-checking. We just rank the passages based on they relevancy to the query.
Sort them from the most relevant to the least."""

    search_query: str = dspy.InputField()
    search_results: list[RelevanceRankedResult] = dspy.InputField()
    reranked_search_results: list[int] = dspy.OutputField()

class WriteSearchQueries(dspy.Signature):
    """Write search queries to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    search_queries: list[str] = dspy.OutputField()

# MIPRO Optimized for FreshstackLangchain
'''
class WriteSearchQueries(dspy.Signature):
    """You are an experienced developer and AI researcher specializing in natural language processing and software development tools. Given a technical question involving langchain, OpenAI APIs, JSON data parsing, or chatbot integration, generate precise and targeted search queries that will help gather relevant information from search engines. Your queries should focus on resolving coding issues, understanding library behaviors, JSON data structures, and best practices for building AI-powered document retrieval and question-answering systems. Ensure queries are clear, specific, and cover potential causes and solutions related to the problem described."""

    question: str = dspy.InputField()
    search_queries: list[str] = dspy.OutputField()
'''

# TODO: Maybe extend to enable multiple filters with one search query
class SearchQueryWithFilter(BaseModel):
    search_query: str
    filter: Optional[str]

class WriteSearchQueriesWithFilters(dspy.Signature):
    """Write search queries with optional filters to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    filters_available: str = dspy.InputField()
    search_queries_with_filters: list[SearchQueryWithFilter] = dspy.OutputField()

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
