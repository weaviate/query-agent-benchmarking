import asyncio
import os
from typing import Dict, Any, Optional
from pathlib import Path

import hashlib
import weaviate

from query_agent_benchmarking.database import create_collection_with_vectorizer, resolve_spec
from query_agent_benchmarking.benchmark_run import _run_eval
from query_agent_benchmarking.utils import load_config, merge_configs, print_results_comparison

def compare_embeddings(
    config_path: Optional[str] = None,
    dataset: Optional[str] = None,
    agent_names: Optional[str | list[str]] = None,
    embedding_models: Optional[list[str]] = None,
    num_trials: Optional[int] = None,
    use_subset: Optional[bool] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_concurrent: Optional[int] = None,
    use_async: Optional[bool] = None,
    agents_host: Optional[str] = None,
    output_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple embedding models by creating temporary collections.
    
    For each embedding model:
    1. Creates a temporary collection with that vectorizer
    2. Runs all specified agents against it
    3. Deletes the temporary collection
    4. Moves to the next embedding model
    
    Returns results keyed by "{agent_name}_{embedding_model}".
    """
    
    if config_path is None:
        config_path = Path(__file__).parent / "benchmark-config.yml"
    
    file_config = load_config(config_path)
    
    # Get agents from parameter or config
    agents = agent_names or file_config.get("agent_name")
    if agents is None:
        raise ValueError("No agent_names provided. Must specify via parameter or in config file.")
    agents = [agents] if isinstance(agents, str) else agents
    
    # Get embedding models from parameter or config
    embedding_models = embedding_models or file_config.get("embedding_models")
    if embedding_models is None:
        raise ValueError(
            "No embedding_models provided. Must specify via parameter or "
            "'embedding_model' list in config file."
        )
    
    # Build override config
    override_config = {
        "dataset": dataset,
        "num_trials": num_trials,
        "use_subset": use_subset,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
        "use_async": use_async,
        "agents_host": agents_host,
        "random_seed": random_seed,
        **kwargs
    }
    
    # Merge base config
    base_config = merge_configs(file_config, override_config)
    
    # Run evaluation for each embedding x agent combination
    all_results = {}
    
    for embedding_model in embedding_models:
        for agent in agents:
            result_key = f"{agent}_{embedding_model}"
            
            # Set agent-specific config
            run_config = base_config.copy()
            run_config["agent_name"] = agent
            
            # Handle output path
            if output_path:
                path = Path(output_path)
                run_config["output_path"] = str(
                    path.parent / f"{path.stem}_{agent}_{embedding_model}{path.suffix}"
                )
            
            # Run evaluation with temporary collection
            try:
                print(f"\n{'#'*60}")
                print(f"# Running: {agent} with {embedding_model}")
                print(f"{'#'*60}\n")
                
                all_results[result_key] = asyncio.run(
                    _run_eval_with_temp_collection(run_config, embedding_model)
                )
            except Exception as e:
                print(f"Error running {result_key}: {str(e)}")
                import traceback
                traceback.print_exc()
                all_results[result_key] = {"error": str(e)}
    
    print_results_comparison(all_results)
    
    return all_results

async def _run_eval_with_temp_collection(
    config: Dict[str, Any], 
    embedding_model: str
) -> Dict[str, Any]:
    """Run evaluation with temporary collection for specified embedding model."""
    
    dataset_name = config.get("dataset")
    
    if not dataset_name:
        raise ValueError("Embedding model comparison only works with built-in datasets")

    # Get the proper collection name from the dataset spec
    spec = resolve_spec(dataset_name)
    alias_collection_name = spec.name_fn(dataset_name)  # e.g., "BrightBiology"
    
    # Create temp collection name and alias
    model_hash = hashlib.md5(embedding_model.encode()).hexdigest()[:8]
    temp_collection_name = f"{alias_collection_name}_{model_hash}"  # e.g., "BrightBiology_abc123"
    default_collection_name = f"{alias_collection_name}_Default"  # e.g., "BrightBiology_Default"
    
    print(f"\n{'='*60}")
    print(f"Creating temporary collection for model: {embedding_model}")
    print(f"Collection name: {temp_collection_name}")
    print(f"Will redirect alias '{alias_collection_name}' to temp collection")
    print(f"{'='*60}\n")
    
    # Create the temporary collection
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    try:
        create_collection_with_vectorizer(
            weaviate_client=weaviate_client,
            dataset_name=dataset_name,
            tag=model_hash,
            embedding_model=embedding_model,
        )
        
        alias_info = weaviate_client.alias.get(alias_name=alias_collection_name)
        if alias_info is None:
            weaviate_client.alias.create(
                alias_name=alias_collection_name,
                new_target_collection=temp_collection_name
            )
        else:
            weaviate_client.alias.update(
                alias_name=alias_collection_name,
                new_target_collection=temp_collection_name
            )
    finally:
        weaviate_client.close()

    try:
        # Just run eval normally - it will use the alias which now points to temp collection
        config_for_eval = config.copy()
        config_for_eval["embeddings"] = [embedding_model]
        
        result = await _run_eval(config_for_eval)
        
        return result
        
    finally:
        # Cleanup: delete temp collection and restore alias
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )
        
        try:
            print(f"\n{'='*60}")
            print(f"Cleaning up temporary collection '{temp_collection_name}'...")
            print(f"{'='*60}\n")
            
            # Restore alias to point back to default collection
            try:
                weaviate_client.alias.update(
                    alias_name=alias_collection_name,
                    new_target_collection=default_collection_name
                )
                print(f"Restored alias '{alias_collection_name}' -> '{default_collection_name}'")
            except Exception as e:
                print(f"Warning: Could not restore alias: {e}")
            
            # Delete temp collection
            if weaviate_client.collections.exists(temp_collection_name):
                weaviate_client.collections.delete(temp_collection_name)
                print(f"Successfully deleted '{temp_collection_name}'\n")
        finally:
            weaviate_client.close()