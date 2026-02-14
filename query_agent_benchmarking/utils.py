import os
import yaml
import re
from typing import Any

import weaviate

from query_agent_benchmarking.models import InMemoryQuery

def get_object_by_dataset_id(dataset_id, objects_list):
    """Retrieve an object by its dataset_id from the objects list."""
    for obj in objects_list:
        if obj["dataset_id"] == dataset_id:
            return obj
    return None

def make_json_serializable(obj):
    """Convert objects to JSON serializable formats."""
    import inspect
    
    if hasattr(obj, '__dict__'):
        # Handle class instances by converting to dict
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_') and not inspect.ismethod(v)}
    elif isinstance(obj, dict):
        # Recursively convert dict values
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert list items
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # These types are already JSON serializable
        return obj
    else:
        # Try to convert to string for other types
        try:
            return str(obj)
        except Exception:
            return f"<Non-serializable object of type {type(obj).__name__}>"
    
def pretty_print_in_memory_query(in_memory_query: InMemoryQuery):
    """
    Pretty prints a dictionary with colored keys and formatted output.
    
    Args:
        dict_to_print: Dictionary to be printed
    """
    print(f"\t\033[96mQuestion\033[0m: {in_memory_query.question}")
    print(f"\t\033[96mDataset IDs\033[0m: {in_memory_query.dataset_ids}")
    print("="*60)
    print("\n\n")

# Note, there isn't a unified document model yet, e.g. InMemoryDocument...
def pretty_print_in_memory_document(in_memory_document_object: dict):
    """
    Pretty prints a dictionary with colored keys and formatted output.
    
    Args:
        dict_to_print: Dictionary to be printed
    """
    print(f"Dataset ID: {in_memory_document_object['dataset_id']}")
    if "content" in in_memory_document_object:
        print(f"\t\033[96mDocument\033[0m: {in_memory_document_object['content']}")
    print("="*60)
    print("\n\n")
    
def pascalize_name(raw: str) -> str:
    # Keep letters/digits, split on non-alnum, PascalCase tokens: "beir/scifact" -> "BeirScifact"
    tokens = re.split(r"[^0-9A-Za-z]+", raw)
    return "".join(t.capitalize() for t in tokens if t)

def add_tag_to_name(original_name: str, tag: str) -> str:
    return f"{original_name}_{tag}"

def load_config(config_path: str):
    if config_path is None or not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config or {}

def merge_configs(file_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    """Merge file-based config with programmatic overrides."""
    merged = file_config.copy()
    
    # Filter out None values from override_config
    filtered_overrides = {k: v for k, v in override_config.items() if v is not None}
    
    # Special handling: if docs_collection is provided, remove dataset from merged config
    if 'docs_collection' in filtered_overrides and 'dataset' in merged:
        del merged['dataset']
    
    # Apply overrides
    merged.update(filtered_overrides)
    
    return merged

def get_provider_headers(provider: str) -> dict[str, str]:
    """
    Get API key headers for third-party embedding providers.
    
    Args:
        provider: The provider name ("cohere", "voyageai", or "weaviate")
    
    Returns:
        Dictionary of headers with API keys from environment variables
    """
    header_map = {
        "cohere": ("X-Cohere-Api-Key", "COHERE_API_KEY"),
        "voyageai": ("X-VoyageAI-Api-Key", "VOYAGEAI_API_KEY"),
    }
    
    if provider not in header_map:
        return {}
    
    header_name, env_var = header_map[provider]
    api_key = os.getenv(env_var)
    
    if not api_key:
        raise ValueError(
            f"Missing API key for provider '{provider}'. "
            f"Please set the {env_var} environment variable."
        )
    
    return {header_name: api_key}

def parse_embedding_model(embedding_model: str) -> tuple[str, str]:
    """Parse embedding model string into provider and model name."""
    if "/" in embedding_model:
        provider, model_name = embedding_model.split("/", 1)
        return provider.lower(), model_name
    return "weaviate", embedding_model
    
def get_weaviate_client(headers: dict[str, str] = None):
    """
    Connect to Weaviate Cloud with optional headers for third-party providers.
    
    Args:
        headers: Optional dictionary of headers (e.g., API keys for Cohere, VoyageAI)
    
    Returns:
        Connected Weaviate client
    """
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        headers=headers or {}
    )

def print_results_comparison(all_results: dict[str, dict[str, Any]]) -> None:
    """Print key metrics for each agent."""
    if not all_results:
        return
    
    # Collect all "_mean" metrics (from aggregated results)
    all_mean_metrics = set()
    for results in all_results.values():
        if "error" not in results:
            for key, value in results.items():
                if key.endswith("_mean") and isinstance(value, (int, float)):
                    all_mean_metrics.add(key)
    
    if not all_mean_metrics:
        return
    
    # Sort metrics by priority
    priority_order = ["recall", "ndcg", "precision", "mrr", "query_time"]
    
    def metric_priority(key):
        for i, term in enumerate(priority_order):
            if term in key.lower():
                return (i, key)
        return (len(priority_order), key)
    
    sorted_metrics = sorted(all_mean_metrics, key=metric_priority)
    
    # Format metric names
    def format_name(key):
        # Remove avg_ and _mean suffixes
        name = key.replace("avg_", "").replace("_mean", "")
        # Handle specific patterns
        name = name.replace("recall_at_", "Recall@").replace("ndcg_at_k", "NDCG@10")
        name = name.replace("query_time", "Time(s)")
        return name
    
    print("\nResults:")
    for agent_name, results in all_results.items():
        print(f"\n{agent_name}:")
        
        if "error" in results:
            print("  ERROR")
            continue
        
        for metric_key in sorted_metrics:
            value = results.get(metric_key)
            if value is not None:
                print(f"  {format_name(metric_key)}: {value:.3f}")
    
    print()