import dspy

from benchmarker.src.dspy_rag.models import SearchResult, SearchQueryWithFilter

class GenerateAnswer(dspy.Signature):
    """Assess the context and answer the question."""

    question: str = dspy.InputField()
    contexts: str = dspy.InputField()
    final_answer: str = dspy.OutputField()

class RerankResults(dspy.Signature):
    """Rerank passages based on their relevance to the query.
    
    IMPORTANT: You must return ONLY THE TOP 5 MOST RELEVANT passage IDs, even if more passages seem relevant.
    
    You are given passages with hybrid retrieval scores that combine:
    - Semantic similarity (how well the meaning matches the query)
    - Lexical matching (keyword/term overlap with the query)
    
    These unified scores provide a strong initial relevance signal. Use them as 
    valuable evidence while applying your deeper understanding to assess:
    - Answer completeness and directness (for questions)
    - Intent satisfaction (for other queries)
    - Information quality and specificity
    
    The hybrid scores already capture both semantic and lexical relevance, but
    you may identify additional relevance factors they miss. The passages may 
    contain contradictions or typos - focus on relevance, not fact-checking.
    
    Remember: You must select and return ONLY the 5 most relevant passage IDs, ranked from most to least relevant.
    Returning more than 5 IDs is not allowed.
    """
    
    query: str = dspy.InputField()
    search_results: list[SearchResult] = dspy.InputField(
        desc="Passages with hybrid scores and initial ranks"
    )
    top_k: int = dspy.InputField(
        desc="Number of passages to return in the reranked list"
    )
    reranked_ids: list[int] = dspy.OutputField(
        desc="EXACTLY 5 passage IDs ordered from most to least relevant. You must return only the top 5 most relevant IDs."
    )

class WriteSearchQueries(dspy.Signature):
    """Write search queries to gather information from a search engine that will help answer the question.
Consider both exploration and result diversity to capture multiple interpretations and facets of a query."""

    question: str = dspy.InputField()
    search_queries: list[str] = dspy.OutputField()

# MIPRO Optimized for FreshstackLangchain
'''
class WriteSearchQueries(dspy.Signature):
    """You are an experienced developer and AI researcher specializing in natural language processing and software development tools. Given a technical question involving langchain, OpenAI APIs, JSON data parsing, or chatbot integration, generate precise and targeted search queries that will help gather relevant information from search engines. Your queries should focus on resolving coding issues, understanding library behaviors, JSON data structures, and best practices for building AI-powered document retrieval and question-answering systems. Ensure queries are clear, specific, and cover potential causes and solutions related to the problem described."""

    question: str = dspy.InputField()
    search_queries: list[str] = dspy.OutputField()
'''

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

# SearchOnlyWithRerankAndSummarize

class SummarizeSearchRelevance(dspy.Signature):
    """Analyze and summarize how a search result addresses the given query.
    
    Evaluate the passage's relevance by considering:
    - How directly it answers or addresses the query
    - The completeness of information provided
    - The specificity and quality of content
    - Whether it contains actionable information
    
    Provide a concise summary (2-3 sentences) explaining:
    1. What relevant information the passage contains
    2. How well it addresses the query's intent
    3. Any limitations or gaps in the information
    """
    
    query: str = dspy.InputField()
    passage: str = dspy.InputField()
    passage_id: int = dspy.InputField(desc="The ID of this passage for reference")
    
    relevance_summary: str = dspy.OutputField(
        desc="A 2-3 sentence summary of how this passage relates to the query and its relevance"
    )
    relevance_score: float = dspy.OutputField(
        desc="A relevance score from 0.0 to 1.0, where 1.0 is perfectly relevant"
    )


class RerankWithSummaries(dspy.Signature):
    """Rerank passages based on their relevance summaries.
    
    You are provided with relevance summaries and scores for each passage.
    Use these summaries to make a final ranking decision.
    
    IMPORTANT: You must return ONLY THE TOP 5 MOST RELEVANT passage IDs.
    
    Consider:
    - The quality and directness of information in each summary
    - The relevance scores as initial guidance
    - How well each passage would satisfy the user's query
    - Prioritize passages that provide complete, actionable answers
    
    Remember: Return EXACTLY 5 passage IDs, ranked from most to least relevant.
    """
    
    query: str = dspy.InputField()
    passage_summaries: list[dict] = dspy.InputField(
        desc="List of dicts with keys: passage_id, relevance_summary, relevance_score"
    )
    top_k: int = dspy.InputField(
        desc="Number of passages to return in the reranked list"
    )
    reranked_ids: list[int] = dspy.OutputField(
        desc="EXACTLY 5 passage IDs ordered from most to least relevant"
    )