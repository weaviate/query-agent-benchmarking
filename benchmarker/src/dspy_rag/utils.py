import os
import yaml
from typing import Optional, Dict, Any
import asyncio

import dspy
import weaviate
from weaviate.classes.query import Filter, Metrics
from weaviate.outputs.query import QueryReturn

from benchmarker.src.dspy_rag.rag_signatures import Source


def weaviate_search_tool(
        query: str,
        collection_name: str,
        target_property_name: str,
        return_dict: bool = False,
        tag_filter_value: Optional[str] | None = None
):
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    collection = weaviate_client.collections.get(collection_name)

    if tag_filter_value:
        search_results = collection.query.hybrid(
            query=query,
            filters=Filter.by_property("tags").contains_any([tag_filter_value]),
            limit=5
        )
    else:
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

async def async_weaviate_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    return_dict: bool = False,
    tag_filter_value: Optional[str] | None = None
):
    async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    await async_client.connect()
    
    try:
        collection = async_client.collections.get(collection_name)
        
        if tag_filter_value:
            search_results = await collection.query.hybrid(
                query=query,
                filters=Filter.by_property("tags").contains_any([tag_filter_value]),
                limit=5
            )
        else:
            search_results = await collection.query.hybrid(
                query=query,
                limit=5
            )
        
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
    
    finally:
        # Always close the connection
        await async_client.close()

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

def get_tag_values(collection_name: str) -> list[str]:
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    collection = weaviate_client.collections.get(collection_name)

    response = collection.aggregate.over_all(
        return_metrics=Metrics("tags").text(
            top_occurrences_count=True,
            top_occurrences_value=True,
            min_occurrences=10
        )
    )
    
    # Extract tag values from top occurrences
    tag_values = [
        occurrence.value 
        for occurrence in response.properties["tags"].top_occurrences
    ]
    return tag_values

def load_optimization_config(config_path: str) -> Dict[str, Any]:
    """
    Load and process configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with processed configuration
        
    Raises:
        ValueError: If file not found or YAML parsing fails
    """
    try:
        # Load YAML configuration
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Apply quick test configuration if enabled
        if config.get("quick_test", {}).get("enabled", False):
            quick_config = config["quick_test"]
            for key, value in quick_config.items():
                if key != "enabled":
                    config[key] = value
            print("Applied quick test configuration")
            
        return config
        
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def setup_dspy():
    """Configure DSPy."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")


def setup_weaviate():
    """Set up Weaviate client connection."""
    cluster_url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not cluster_url or not api_key:
        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY environment variables are required")
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key),
    )
    
    print(f"Connected to Weaviate cluster: {cluster_url}")
    return client


def print_configuration_summary(config: Dict[str, Any]):
    """Print a summary of the optimization configuration."""
    print("=" * 60)
    print("DSPy Optimization Configuration")
    print("=" * 60)
    print(f"Dataset: {config['dataset_name']}")
    print(f"Agent: {config['agent_name']}")
    print(f"Optimizer: {config['optimizer_type']}")
    print(f"Metric: {config['metric_type']}")
    print(f"Output Directory: {config['output_dir']}")
    
    if config.get("quick_test", {}).get("enabled"):
        print("Quick Test Mode: ENABLED")
    
    print("=" * 60)


async def test_search_functions():
    """Simplified test for both sync and async search functions."""
    print("\n" + "=" * 50)
    print("Testing Search Functions")
    print("=" * 50)
    
    # Test parameters
    test_query = "What is LangChain?"
    collection_name = "FreshstackLangchain"
    target_property_name = "docs_text"
    
    print(f"Query: {test_query}")
    print(f"Collection: {collection_name}")
    
    try:
        # Test sync search
        print("\n--- Sync Search ---")
        sync_contexts, sync_sources = weaviate_search_tool(
            query=test_query,
            collection_name=collection_name,
            target_property_name=target_property_name,
            return_dict=False
        )
        print(f"✓ Sync search successful: {len(sync_sources)} results")
        
        # Test async search
        print("\n--- Async Search ---")
        async_contexts, async_sources = await async_weaviate_search_tool(
            query=test_query,
            collection_name=collection_name,
            target_property_name=target_property_name,
            return_dict=False
        )
        print(f"✓ Async search successful: {len(async_sources)} results")
        
        # Quick comparison
        results_match = len(sync_sources) == len(async_sources)
        print(f"✓ Results consistency: {'PASS' if results_match else 'FAIL'}")
        
    except Exception as e:
        print(f"✗ Search test failed: {type(e).__name__}: {e}")


def test_get_tag_values():
    """Test the get_tag_values function."""
    print("\n" + "=" * 50)
    print("Testing get_tag_values Function")
    print("=" * 50)
    
    collection_name = "FreshstackLangchain"  # Update this to match your collection
    
    print(f"Getting tag values for collection: {collection_name}")
    
    try:
        print("\n--- Tag Values ---")
        tag_values = get_tag_values(collection_name)
        print("✓ get_tag_values function executed successfully")
        print(tag_values)
            
    except Exception as e:
        print(f"✗ get_tag_values test failed: {type(e).__name__}: {e}")


def main():
    """Main function to run simplified tests."""
    # Check if required environment variables are set
    required_env_vars = ["WEAVIATE_URL", "WEAVIATE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these environment variables before running the test.")
        return
    
    print("Starting simplified tests...")
    
    test_get_tag_values()
    
    # Test search functions (includes async)
    asyncio.run(test_search_functions())
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()