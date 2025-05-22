from pydantic import BaseModel
from typing import Optional, Dict, Any
import dspy

class Source(BaseModel):
    object_id: str

class AgentRAGResponse(BaseModel):
    final_answer: str
    sources: list[Source]
    usage: Optional[Dict[str, Any]] = None

class GenerateAnswer(dspy.Signature):
    """Assess the context and answer the question."""

    question: str = dspy.InputField()
    contexts: str = dspy.InputField()
    final_answer: str = dspy.OutputField()

class WriteSearchQueries(dspy.Signature):
    """Write search queries to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    search_queries: list[str] = dspy.OutputField()

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
