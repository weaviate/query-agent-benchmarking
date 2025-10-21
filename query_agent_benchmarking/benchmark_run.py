import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import yaml

import weaviate

from query_agent_benchmarking.agent import AgentBuilder
from query_agent_benchmarking.dataset import (
    in_memory_dataset_loader,
    load_queries_from_weaviate_collection,
)
from query_agent_benchmarking.models import (
    DocsCollection,
    QueriesCollection,
    InMemoryQuery,
)
from query_agent_benchmarking.query_agent_benchmark import (
    run_queries,
    run_queries_async,
    analyze_results,
    aggregate_metrics
)
from query_agent_benchmarking.utils import pretty_print_in_memory_query
from query_agent_benchmarking.config import supported_datasets


DEFAULT_CONFIG_PATH = Path(__file__).parent / "benchmark-config.yml"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(file_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
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


async def _run_eval_async(config: Dict[str, Any]) -> Dict[str, Any]:
    agents_host = config.get("agents_host", "https://api.agents.weaviate.io")
    use_async = config.get("use_async", True)
    
    # Determine if using built-in dataset or custom collection
    dataset_name = config.get("dataset")
    docs_collection = config.get("docs_collection")
    queries_input = config.get("queries")
    
    if dataset_name:
        # Built-in dataset paths
        if dataset_name not in supported_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not supported. "
                f"Supported datasets are: {supported_datasets}"
            )
        
        _, queries = in_memory_dataset_loader(dataset_name)
        dataset_identifier = dataset_name
        
    elif docs_collection and queries_input:
        # Custom collection path
        if not isinstance(docs_collection, DocsCollection):
            raise ValueError("docs_collection must be a DocsCollection object")
        
        # Load queries based on the type of queries_input
        if isinstance(queries_input, QueriesCollection):
            # Load from Weaviate
            queries = load_queries_from_weaviate_collection(
                collection_name=queries_input.collection_name,
                query_content_key=queries_input.query_content_key,
                gold_ids_key=queries_input.gold_ids_key,
            )
        elif isinstance(queries_input, list):
            # Verify all items are InMemoryQuery
            if not queries_input:
                raise ValueError("Queries list cannot be empty")
            if not all(isinstance(q, InMemoryQuery) for q in queries_input):
                raise ValueError(
                    "All queries must be InMemoryQuery objects. "
                    f"Found: {set(type(q).__name__ for q in queries_input)}"
                )
            queries = queries_input
        else:
            raise ValueError(
                f"Queries must be either QueriesCollection or List[InMemoryQuery]. "
                f"Got: {type(queries_input)}"
            )
        
        dataset_identifier = docs_collection.collection_name
        
    else:
        raise ValueError(
            "Must provide either 'dataset' (for built-in) or "
            "'docs_collection' + 'queries' (for custom)"
        )
    
    print(f"There are \033[92m{len(queries)}\033[0m total queries in this dataset.\n")
    print("\033[92mFirst Query\033[0m")
    pretty_print_in_memory_query(queries[0])

    # Handle subset if requested
    if config.get("use_subset", False):
        import random
        random.seed(config.get("random_seed", 24))
        random.shuffle(queries)
        queries = queries[:config["num_samples"]]
        print(f"Using a subset of {config['num_samples']} queries.")

    # Build agent
    if dataset_name:
        query_agent = AgentBuilder(
            agent_name=config["agent_name"],
            dataset_name=dataset_name,
            agents_host=agents_host,
            use_async=use_async,
        )
    else:
        query_agent = AgentBuilder(
            agent_name=config["agent_name"],
            docs_collection=docs_collection,
            agents_host=agents_host,
            use_async=use_async,
        )

    num_trials = config.get("num_trials", 1)
    metrics_across_trials = []

    # Run trials
    for trial in range(num_trials):
        print(f"\033[92mRunning trial {trial+1}/{num_trials}\033[0m")

        if use_async:        
            print("\033[92mRunning queries async!\033[0m")
            await query_agent.initialize_async()
            
            try:
                results = await run_queries_async(
                    queries=queries,
                    query_agent=query_agent,
                    batch_size=config.get("batch_size", 10),
                    max_concurrent=config.get("max_concurrent", 5)
                )
            finally:
                await query_agent.close_async()
        else:
            print("\n\033[94mRunning synchronous benchmark\033[0m")
            results = run_queries(
                queries=queries,
                query_agent=query_agent,
            )

        # Analyze results
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )

        metrics = await analyze_results(
            results=results,
            ground_truths=queries,
            dataset_name=dataset_name, 
        )
        print(metrics)

        weaviate_client.close()
        metrics_across_trials.append(metrics)

    # Aggregate and save results
    aggregated_metrics = aggregate_metrics(metrics_across_trials)
    aggregated_metrics["timestamp"] = datetime.now().isoformat()
    
    # Save results
    output_path = config.get("output_path")
    if output_path is None:
        dataset_name_for_file = dataset_identifier.replace("/", "-")
        output_path = f"{dataset_name_for_file}-{config['agent_name']}-{num_trials}-results.json"
    
    with open(output_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    print(f"\n\033[92mResults saved to {output_path}\033[0m")
    
    return aggregated_metrics


def run_eval(
    config_path: Optional[str] = None,
    dataset: Optional[str] = None,
    docs_collection: Optional[DocsCollection] = None,
    queries: Optional[Union[QueriesCollection, List[InMemoryQuery]]] = None,
    agent_name: Optional[str] = None,
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
) -> Dict[str, Any]:
    """
    Run evaluation benchmark for query agents.
    
    Works with both built-in datasets and custom collections.
    
    Args:
        config_path: Path to YAML config file (default: benchmark-config.yml in package dir)
        
        # Built-in dataset mode:
        dataset: Dataset name (e.g., "bright/biology")
        
        # Custom collection mode:
        docs_collection: DocsCollection object specifying the document collection
        queries: Queries in one of two formats:
            - QueriesCollection: Loads from Weaviate
            - List[InMemoryQuery]: Pre-built query objects
        
        # Common parameters:
        agent_name: Name of the agent to benchmark
        num_trials: Number of trials to run
        use_subset: Whether to use a subset of queries
        num_samples: Number of samples to use if use_subset=True
        batch_size: Batch size for async queries
        max_concurrent: Max concurrent requests for async queries
        use_async: Whether to use async mode
        agents_host: URL of the agents host
        output_path: Custom path for output JSON file
        random_seed: Random seed for subset selection
        **kwargs: Additional config parameters
    
    Returns:
        Dict containing aggregated metrics
    
    Examples:
        # Built-in dataset with default config
        >>> run_eval()
        
        # Built-in dataset with overrides
        >>> run_eval(dataset="bright/biology", num_trials=3)
        
        # Custom collection with Weaviate queries
        >>> docs = DocsCollection(name="MyDocs", id_key="id")
        >>> queries = QueriesCollection(
        ...     name="MyQueries",
        ...     id_key="id"
        ... )
        >>> run_eval(docs_collection=docs, queries=queries, agent_name="my-agent")
        
        # Custom collection with in-memory queries
        >>> docs = DocsCollection(name="MyDocs")
        >>> queries = [
        ...     InMemoryQuery(question="What is X?", query_id="q1", dataset_ids=["doc1"]),
        ...     InMemoryQuery(question="Explain Y", query_id="q2", dataset_ids=["doc2"])
        ... ]
        >>> run_eval(docs_collection=docs, queries=queries, agent_name="my-agent")
    """
    
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Load base config from file
    file_config = load_config(config_path)
    
    # Build override config from parameters
    override_config = {
        "dataset": dataset,
        "docs_collection": docs_collection,
        "queries": queries,
        "agent_name": agent_name,
        "num_trials": num_trials,
        "use_subset": use_subset,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
        "use_async": use_async,
        "agents_host": agents_host,
        "output_path": output_path,
        "random_seed": random_seed,
        **kwargs
    }
    
    # Merge configs
    final_config = merge_configs(file_config, override_config)
    
    # Run evaluation
    return asyncio.run(_run_eval_async(final_config))