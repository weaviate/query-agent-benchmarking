"""
Search benchmark runner - evaluates ranked retrieval performance.

This module provides the entry point for running search mode benchmarks,
which measure how well the system retrieves relevant documents using
IR metrics like Recall@K, nDCG@K, etc.
"""

import asyncio
from pathlib import Path
from typing import Optional, Any, Union

from query_agent_benchmarking.agent import SearchAgentBuilder
from query_agent_benchmarking.dataset import (
    in_memory_dataset_loader,
    load_queries_from_weaviate_collection,
)
from query_agent_benchmarking.models import (
    DocsCollection,
    QueriesCollection,
    InMemoryQuery,
    InMemorySearchQuery,
)
from query_agent_benchmarking.query_agent_benchmark import (
    run_search_queries,
    run_search_queries_async,
    analyze_search_results,
    aggregate_metrics
)
from query_agent_benchmarking.result_serialization import (
    save_trial_results,
    save_trial_metrics,
    save_aggregated_results,
)
from query_agent_benchmarking.utils import (
    pretty_print_in_memory_query, 
    load_config, 
    merge_configs,
    print_results_comparison,
)
from query_agent_benchmarking.config import (
    supported_search_datasets,
    resolve_named_vector_target,
    resolve_embedding_providers,
)


DEFAULT_CONFIG_PATH = Path(__file__).parent / "benchmark-config.yml"


def _resolve_search_agent_name(
    config: dict[str, Any],
    dataset_identifier: Optional[str],
) -> str:
    """Resolve agent name and optional target vector from config."""
    raw_agent_name = config.get("search_agent_name") or config.get("agent_name")
    if raw_agent_name is None:
        raise ValueError("No search_agent_name provided in config")
    if isinstance(raw_agent_name, list):
        raise ValueError(
            "run_search_eval expects a single agent name. "
            "Use run_search_evals for multiple agents."
        )

    target_vector = config.get("search_target")
    if target_vector is None:
        # Backward-compatible fallback for legacy configs.
        target_vector = config.get("search_target_vector")
    if target_vector is None:
        target_vector = config.get("target_vector")

    if "[" in raw_agent_name:
        if target_vector:
            raise ValueError(
                "Specify target vector either via search_agent_name suffix "
                "(e.g., hybrid-search[text_content_weaviate]) or "
                "search_target/search_target_vector, not both."
            )
        base_name, inline_target = SearchAgentBuilder._parse_agent_name(raw_agent_name)
        resolved_inline_target = resolve_named_vector_target(
            dataset_name=dataset_identifier,
            target_vector=inline_target,
        )
        return f"{base_name}[{resolved_inline_target}]"

    if target_vector:
        resolved_target_vector = resolve_named_vector_target(
            dataset_name=dataset_identifier,
            target_vector=target_vector,
        )
        raw_agent_name = f"{raw_agent_name}[{resolved_target_vector}]"

    return raw_agent_name


def _resolve_embedding_models(config: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve text/image embedding model config with backward compatibility.

    Legacy `embedding_model` is treated as `text_embedding_model`.
    """
    text_embedding_model = config.get("text_embedding_model")
    if text_embedding_model is None:
        text_embedding_model = config.get("embedding_model")
    image_embedding_model = config.get("image_embedding_model")
    return text_embedding_model, image_embedding_model


async def _run_search_eval(config: dict[str, Any]) -> dict[str, Any]:
    """Internal async implementation for search evaluation."""
    agents_host = config.get("agents_host", "https://api.agents.weaviate.io")
    use_async = config.get("use_async", True)
    
    # Determine if using built-in dataset or custom collection
    dataset_name = config.get("search_dataset")
    docs_collection = config.get("docs_collection")
    queries_input = config.get("queries")

    if dataset_name:
        # Built-in dataset paths
        if dataset_name not in supported_search_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not supported. "
                f"Supported datasets are: {supported_search_datasets}"
            )

        _, queries = in_memory_dataset_loader(dataset_name, queries_only=True)
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
            # Verify all items are InMemoryQuery or InMemorySearchQuery
            if not queries_input:
                raise ValueError("Queries list cannot be empty")
            if not all(isinstance(q, (InMemoryQuery, InMemorySearchQuery)) for q in queries_input):
                raise ValueError(
                    "All queries must be InMemoryQuery or InMemorySearchQuery objects. "
                    f"Found: {set(type(q).__name__ for q in queries_input)}"
                )
            queries = queries_input
        else:
            raise ValueError(
                f"Queries must be either QueriesCollection or List[InMemoryQuery/InMemorySearchQuery]. "
                f"Got: {type(queries_input)}"
            )
        
        dataset_identifier = docs_collection.collection_name
        
    else:
        raise ValueError(
            "Must provide either 'search_dataset' (for built-in) or "
            "'docs_collection' + 'queries' (for custom)"
        )
    
    config["dataset_identifier"] = dataset_identifier

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
    text_embedding_model, image_embedding_model = _resolve_embedding_models(config)
    agent_name = _resolve_search_agent_name(
        config,
        dataset_identifier=dataset_identifier,
    )
    _, resolved_target_vector = SearchAgentBuilder._parse_agent_name(agent_name)
    embedding_providers = resolve_embedding_providers(
        dataset_name=dataset_identifier,
        target_vector=resolved_target_vector,
        embedding_providers=config.get("embedding_providers"),
        embedding_models=[
            text_embedding_model,
            image_embedding_model,
            config.get("embedding_model"),
        ],
    )

    # Persist normalized config keys for downstream serialization.
    config["search_agent_name"] = agent_name
    config["agent_name"] = agent_name
    if text_embedding_model is not None:
        config["text_embedding_model"] = text_embedding_model
        config["embedding_model"] = text_embedding_model
    if image_embedding_model is not None:
        config["image_embedding_model"] = image_embedding_model
    if embedding_providers:
        config["embedding_providers"] = embedding_providers
    
    external_service_host = config.get("external_service_host")
    
    if dataset_name:
        query_agent = SearchAgentBuilder(
            agent_name=agent_name,
            dataset_name=dataset_name,
            agents_host=agents_host,
            use_async=use_async,
            embedding_model=text_embedding_model,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            embedding_providers=embedding_providers,
            external_service_host=external_service_host,
        )
    else:
        query_agent = SearchAgentBuilder(
            agent_name=agent_name,
            docs_collection=docs_collection,
            agents_host=agents_host,
            use_async=use_async,
            embedding_model=text_embedding_model,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            embedding_providers=embedding_providers,
            external_service_host=external_service_host,
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
                results = await run_search_queries_async(
                    queries=queries,
                    query_agent=query_agent,
                    batch_size=config.get("batch_size", 10),
                    max_concurrent=config.get("max_concurrent", 5)
                )
            finally:
                await query_agent.close_async()
        else:
            print("\n\033[94mRunning synchronous benchmark\033[0m")
            results = run_search_queries(
                queries=queries,
                query_agent=query_agent,
            )

        save_trial_results(
            results=results,
            config=config,
            trial_number=trial+1,
        )

        # Analyze results
        metrics = await analyze_search_results(
            results=results,
            ground_truths=queries,
            dataset_name=dataset_name,
        )
        print(metrics)
        save_trial_metrics(
            metrics=metrics,
            config=config,
            trial_number=trial+1,
        )
        metrics_across_trials.append(metrics)

    # Aggregate and save results
    aggregated_metrics = aggregate_metrics(metrics_across_trials)
    save_aggregated_results(
        aggregated_metrics=aggregated_metrics,
        config=config,
    )
    return aggregated_metrics


def run_search_eval(
    config_path: Optional[str] = None,
    search_dataset: Optional[str] = None,
    docs_collection: Optional[DocsCollection] = None,
    queries: Optional[Union[QueriesCollection, list[InMemoryQuery], list[InMemorySearchQuery]]] = None,
    agent_name: Optional[str] = None,
    num_trials: Optional[int] = None,
    use_subset: Optional[bool] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_concurrent: Optional[int] = None,
    use_async: Optional[bool] = None,
    agents_host: Optional[str] = None,
    external_service_host: Optional[str] = None,
    output_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    embedding_model: Optional[str] = None,
    text_embedding_model: Optional[str] = None,
    image_embedding_model: Optional[str] = None,
    embedding_providers: Optional[Union[str, list[str]]] = None,
    search_target: Optional[str] = None,
    search_target_vector: Optional[str] = None,
    **kwargs
) -> dict[str, Any]:
    """
    Run a search benchmark evaluation.

    Evaluates ranked retrieval performance using IR metrics like Recall@K, nDCG@K.

    Args:
        config_path: Path to YAML config file. Defaults to built-in config.
        search_dataset: Name of built-in dataset (e.g., "beir/scifact", "enron").
        docs_collection: DocsCollection for custom datasets.
        queries: Queries as QueriesCollection, list[InMemoryQuery], or list[InMemorySearchQuery].
        agent_name: Agent to use ("query-agent-search-only", "hybrid-search", or "external_service").
        num_trials: Number of evaluation trials to run.
        use_subset: Whether to use a random subset of queries.
        num_samples: Number of samples if use_subset is True.
        batch_size: Batch size for async processing.
        max_concurrent: Max concurrent requests for async.
        use_async: Whether to use async execution.
        agents_host: Host URL for the agents service.
        external_service_host: Host URL for external mode (e.g., "http://localhost:8000").
        output_path: Path to save results.
        random_seed: Random seed for reproducibility.
        embedding_model: Deprecated alias for text_embedding_model.
        text_embedding_model: Text embedding model for search query vectorization.
        image_embedding_model: Image/multimodal embedding model for named image vectors.
        embedding_providers: Provider list for header injection (e.g. ["cohere", "voyageai"])
            or "auto" to infer from target vectors.
        search_target: Canonical named vector(s), e.g. "text_content_weaviate",
            "image_content_weaviate", or "text_content_weaviate+image_content_weaviate".
        search_target_vector: Legacy vector-name key. Prefer search_target.
        **kwargs: Additional config overrides.
        
    Returns:
        Dict containing aggregated metrics across trials.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Load base config from file
    file_config = load_config(config_path)
    
    # Build override config from parameters
    override_config = {
        "search_dataset": search_dataset,
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
        "external_service_host": external_service_host,
        "output_path": output_path,
        "random_seed": random_seed,
        "embedding_model": embedding_model,
        "text_embedding_model": text_embedding_model,
        "image_embedding_model": image_embedding_model,
        "embedding_providers": embedding_providers,
        "search_target": search_target,
        "search_target_vector": search_target_vector,
        **kwargs
    }

    # Merge configs
    final_config = merge_configs(file_config, override_config)

    # Run evaluation
    return asyncio.run(_run_search_eval(final_config))


def run_search_evals(
    config_path: Optional[str] = None,
    search_dataset: Optional[str] = None,
    docs_collection: Optional[DocsCollection] = None,
    queries: Optional[Union[QueriesCollection, list[InMemoryQuery], list[InMemorySearchQuery]]] = None,
    agent_names: Optional[Union[str, list[str]]] = None,
    num_trials: Optional[int] = None,
    use_subset: Optional[bool] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_concurrent: Optional[int] = None,
    use_async: Optional[bool] = None,
    agents_host: Optional[str] = None,
    external_service_host: Optional[str] = None,
    output_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    embedding_model: Optional[str] = None,
    text_embedding_model: Optional[str] = None,
    image_embedding_model: Optional[str] = None,
    embedding_providers: Optional[Union[str, list[str]]] = None,
    search_target: Optional[str] = None,
    search_target_vector: Optional[str] = None,
    **kwargs
) -> dict[str, dict[str, Any]]:
    """Run search benchmark for multiple query agents and compare results."""
    
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    file_config = load_config(config_path)
    
    # Get agents from parameter or config
    agents = agent_names or file_config.get("search_agent_name") or file_config.get("agent_name")
    if agents is None:
        raise ValueError("No agent_names provided. Must specify via parameter or in config file (search_agent_name).")
    
    # Convert to list if string
    agents = [agents] if isinstance(agents, str) else agents
    
    # Build override config once (shared by all agents except agent_name)
    override_config = {
        "search_dataset": search_dataset,
        "docs_collection": docs_collection,
        "queries": queries,
        "num_trials": num_trials,
        "use_subset": use_subset,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
        "use_async": use_async,
        "agents_host": agents_host,
        "external_service_host": external_service_host,
        "random_seed": random_seed,
        "embedding_model": embedding_model,
        "text_embedding_model": text_embedding_model,
        "image_embedding_model": image_embedding_model,
        "embedding_providers": embedding_providers,
        "search_target": search_target,
        "search_target_vector": search_target_vector,
        **kwargs
    }
    
    # Run evaluation for each agent
    all_results = {}
    for agent in agents:
        # Set agent-specific overrides
        agent_override = override_config.copy()
        agent_override["agent_name"] = agent
        
        # Handle output path
        if output_path:
            path = Path(output_path)
            agent_override["output_path"] = str(path.parent / f"{path.stem}_{agent}{path.suffix}")
        
        # Run evaluation
        try:
            all_results[agent] = asyncio.run(_run_search_eval(merge_configs(file_config, agent_override)))
        except Exception as e:
            all_results[agent] = {"error": str(e)}
    
    print_results_comparison(all_results)

    return all_results
