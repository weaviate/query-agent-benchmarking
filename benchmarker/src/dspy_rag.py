import os
from typing import Optional, Dict, Any

import dspy
from pydantic import BaseModel
import weaviate
from weaviate.outputs.query import QueryReturn

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

def weaviate_search_tool(
        query: str,
        collection_name: str,
        target_property_name: str,
        return_dict: bool = False
):
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    collection = weaviate_client.collections.get(collection_name)

    search_results = collection.query.hybrid(
        query=query,
        limit=5
    )

    weaviate_client.close()

    object_ids = []
    if search_results.objects:
        for obj in search_results.objects:
            object_ids.append(Source(
                object_id=str(obj.uuid)
            ))

    if return_dict:
        # Return dictionary with numeric IDs (1-based) and maintain mapping to UUIDs
        return _dictify_search_results(search_results, view_properties=[target_property_name]), object_ids
    else:
        # Return traditional string format
        return _stringify_search_results(search_results, view_properties=[target_property_name]), object_ids

def _stringify_search_results(search_results: QueryReturn, view_properties=None) -> str:
    """
    Convert Weaviate search results to a readable string format.
    
    Args:
        search_results: The QueryReturn object from Weaviate
        view_properties: List of property names to include (None means include nothing)
                         Can include metadata fields prefixed with underscore
    
    Returns:
        A formatted string representation of the search results
    """
    result_str = f"Found {len(search_results.objects)} results:\n\n"
    
    for i, obj in enumerate(search_results.objects):
        result_str += f"Result {i+1}:\n"
        
        if view_properties:
            if obj.properties:
                properties_to_show = {k: v for k, v in obj.properties.items() if k in view_properties}
                
                if properties_to_show:
                    result_str += "Properties:\n"
                    for key, value in properties_to_show.items():
                        result_str += f"  {key}: {value}\n"
            
            if obj.metadata:
                metadata_fields = []
                for attr in dir(obj.metadata):
                    if attr in view_properties:
                        value = getattr(obj.metadata, attr)
                        if value is not None:
                            metadata_fields.append((attr, value))
                
                if metadata_fields:
                    result_str += "Metadata:\n"
                    for attr, value in metadata_fields:
                        result_str += f"  {attr}: {value}\n"
        
        result_str += "\n"
    
    return result_str

def _dictify_search_results(search_results: QueryReturn, view_properties=None) -> dict[int, str]:
    """
    Convert Weaviate search results to a dictionary with integer keys (1-based).
    
    Args:
        search_results: The QueryReturn object from Weaviate
        view_properties: List of property names to include
    
    Returns:
        A dictionary mapping numeric IDs to formatted search result strings
    """
    result_dict = {}
    
    for i, obj in enumerate(search_results.objects):
        result_id = i + 1  # 1-based indexing
        result_str = f"Result {result_id}:\n"
        
        if view_properties:
            if obj.properties:
                properties_to_show = {k: v for k, v in obj.properties.items() if k in view_properties}
                
                if properties_to_show:
                    result_str += "Properties:\n"
                    for key, value in properties_to_show.items():
                        result_str += f"  {key}: {value}\n"
            
            if obj.metadata:
                metadata_fields = []
                for attr in dir(obj.metadata):
                    if attr in view_properties:
                        value = getattr(obj.metadata, attr)
                        if value is not None:
                            metadata_fields.append((attr, value))
                
                if metadata_fields:
                    result_str += "Metadata:\n"
                    for attr, value in metadata_fields:
                        result_str += f"  {attr}: {value}\n"
        
        result_dict[result_id] = result_str
    
    return result_dict

class RAGAblation():
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        write_queries: Optional[bool] = False,
        filter_results: Optional[bool] = False,
        summarize_results: Optional[bool] = False
    ):
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.query_writer = dspy.Predict(WriteSearchQueries)
        self.result_filter = dspy.Predict(FilterIrrelevantSearchResults)
        self.result_summarizer = dspy.Predict(SummarizeSearchResults)
        self.collection_name = collection_name
        self.target_property_name = target_property_name
        self.write_queries = write_queries
        self.filter_results = filter_results
        self.summarize_results = summarize_results

    def _merge_usage(self, *usages):
        """Utility: add up DSPy usage dicts."""
        merged = {}
        for usage in usages:
            if not usage:
                continue
            for lm_id, stats in usage.items():
                m = merged.setdefault(lm_id, {"prompt_tokens": 0,
                                              "completion_tokens": 0})
                m["prompt_tokens"]    += stats.get("prompt_tokens", 0)
                m["completion_tokens"] += stats.get("completion_tokens", 0)
        return merged

    def forward(self, question):
        query_pred = None
        if self.write_queries:
            query_pred = self.query_writer(question=question)
            questions = query_pred.search_queries
            print(f"\033[96mOriginal Question: {question}\033[0m")
            print(f"\033[33mSearch Queries: {questions}\033[0m")
            
            # Debug: check if usage tracking is working
            if query_pred.get_lm_usage():
                print(f"\033[94mQuery writer usage: {query_pred.get_lm_usage()}\033[0m")
            else:
                print("\033[91mWARNING: No usage data captured for query writer\033[0m")
        else:
            questions = [question]

        contexts, sources = [], []
        all_preds = []
        if query_pred:
            all_preds.append(query_pred)
            
        # Process each query individually through the pipeline
        for q in questions:
            # Determine return format based on whether filtering is enabled
            return_dict = self.filter_results or self.summarize_results
            
            # Get search results (as dict if filtering/summarization is enabled)
            result, src = weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                return_dict=return_dict
            )
            
            # Apply filtering if enabled
            if self.filter_results and result:
                filter_pred = self.result_filter(
                    question=q,
                    search_results=result  # Already a dict[int, str]
                )
                all_preds.append(filter_pred)
                
                # Debug: check filter usage
                if filter_pred.get_lm_usage():
                    print(f"\033[94mFilter usage for query '{q}': {filter_pred.get_lm_usage()}\033[0m")
                else:
                    print("\033[91mWARNING: No usage data captured for filter\033[0m")
                
                # Keep only the filtered results
                filtered_result = {}
                for result_id in filter_pred.filtered_results:
                    if result_id in result:
                        filtered_result[result_id] = result[result_id]
                
                result = filtered_result
                print(f"\033[33mFiltered Results Applied for query: {q} - Kept IDs: {filter_pred.filtered_results}\033[0m")
            
            # Apply summarization if enabled
            if self.summarize_results and result:
                # If we have a dict but didn't filter, use all results
                summarize_pred = self.result_summarizer(
                    question=q,
                    search_results=result  # Could be filtered dict or original dict
                )
                
                # Debug: check summarizer usage
                if summarize_pred.get_lm_usage():
                    print(f"\033[94mSummarizer usage for query '{q}': {summarize_pred.get_lm_usage()}\033[0m")
                else:
                    print("\033[91mWARNING: No usage data captured for summarizer\033[0m")
                    
                result = summarize_pred.summary  # Now a string
                all_preds.append(summarize_pred)
                print(f"\033[33mSummarization Applied for query: {q}\033[0m")
            
            # Convert dict to string if necessary
            if isinstance(result, dict):
                result_str = ""
                for id_num, content in result.items():
                    result_str += f"{content}\n"
                contexts.append(result_str)
            else:
                # Already a string (either from original or after summarization)
                contexts.append(result)
            
            sources.extend(src)

        # Final answer generation
        answer_pred = self.generate_answer(
            question=question,  # Use the original question
            contexts="".join(contexts)
        )
        all_preds.append(answer_pred)
        
        # Debug: check answer generator usage
        if answer_pred.get_lm_usage():
            print(f"\033[94mAnswer generator usage: {answer_pred.get_lm_usage()}\033[0m")
        else:
            print("\033[91mWARNING: No usage data captured for answer generator\033[0m")

        # Check individual usages before merging
        for i, pred in enumerate(all_preds):
            if pred is not None:
                usage = pred.get_lm_usage()
                print(f"\033[95mPrediction {i} usage: {usage}\033[0m")

        # Merge usage from all predictors
        total_usage = self._merge_usage(*[pred.get_lm_usage() for pred in all_preds if pred])
        print(f"\033[95mMerged total usage: {total_usage}\033[0m")

        # Create a response object with the expected structure
        response = AgentRAGResponse(
            final_answer=answer_pred.final_answer,
            sources=sources,
            usage=total_usage  # Set usage during initialization
        )

        return response

def main():
    """Test the VanillaRAG implementation."""
    print("\033[95mSetting up DSPy with usage tracking...\033[0m")
    # Make sure to create the LM instance with the right settings
    lm = dspy.LM(
        'openai/gpt-4o', 
        api_key=os.getenv("OPENAI_API_KEY"),
        cache=False,
    )
    
    # Explicitly configure DSPy with track_usage=True
    dspy.configure(lm=lm, track_usage=True)
    print(f"\033[95mDSPy configured with: {lm}\033[0m")
    
    collection_name = "WixKB"
    target_property_name = "contents"
    rag = RAGAblation(
        collection_name, 
        target_property_name, 
        write_queries=True,
        filter_results=True,  # Enable filtering for testing
        summarize_results=False
    )
    
    test_question = "How does Wix mobile work?"
    print(f"\033[96mQuestion: {test_question}\033[0m")
    
    response = rag.forward(test_question)
    print(f"\033[92m{response.final_answer}\033[0m\n")
    print("\033[96mSources:\033[0m\n")
    print(f"\033[92m{response.sources}\033[0m")

    print("\033[96mUsage:\033[0m")
    print(f"\033[92m{response.usage}\033[0m")

if __name__ == "__main__":
    main()
