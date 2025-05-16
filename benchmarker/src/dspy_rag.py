import os

import dspy
import weaviate
from weaviate.outputs.query import QueryReturn

class GenerateAnswer(dspy.Signature):
    """Assess the context and answer the question."""

    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField()

def weaviate_search_tool(
        query: str,
        collection_name: str,
        target_property_name: str
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

    return _stringify_search_results(search_results)

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

class VanillaRAG():
    def __init__(self, collection_name: str, target_property_name: str):
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.collection_name = collection_name
        self.target_property_name = target_property_name

    def forward(self, question):
        contexts = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name
        )
        return self.generate_answer(
            question=question,
            contexts=contexts
        ).answer

def main():
    """Test the VanillaRAG implementation."""
    lm = dspy.LM('openai/gpt-4o', api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)
    
    collection_name = "WixKB"
    target_property_name = "contents"
    rag = VanillaRAG(collection_name, target_property_name)
    
    test_question = "What is Wix?"
    print(f"\033[96mQuestion: {test_question}\033[0m")
    
    answer = rag.forward(test_question)
    print(f"\033[92mAnswer: {answer}\033[0m")

if __name__ == "__main__":
    main()
