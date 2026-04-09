"""
Ask benchmark runner - evaluates question answering performance.

This module provides the entry point for running ask mode benchmarks,
which measure how well the system answers questions using LLM-as-judge
evaluation for semantic alignment.
"""

import asyncio
from pathlib import Path
from typing import Optional, Any, Union

from query_agent_benchmarking.agent import AskAgentBuilder, EngramDSPyAgent
from query_agent_benchmarking.models import (
    DocsCollection,
    AskQueriesCollection,
    InMemoryAskQuery,
)
from query_agent_benchmarking.dataset import (
    in_memory_ask_dataset_loader,
    load_ask_queries_from_weaviate,
)
from query_agent_benchmarking.query_agent_benchmark import (
    run_ask_queries,
    run_ask_queries_async,
    analyze_ask_results,
    aggregate_metrics
)
from query_agent_benchmarking.result_serialization import (
    save_trial_metrics,
    save_ask_trial_results,
    save_aggregated_results,
)
from query_agent_benchmarking.utils import (
    load_config, 
    merge_configs,
)
from query_agent_benchmarking.config import supported_ask_datasets
from query_agent_benchmarking.qa_system_prompt_registry import get_system_prompt


DEFAULT_CONFIG_PATH = Path(__file__).parent / "benchmark-config.yml"
DEFAULT_AGENT_CONFIG_PATH = Path(__file__).parent / "agent-config.yml"


def _load_agent_config(agent_name: str, agent_config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load agent-specific parameters from agent-config.yml."""
    path = agent_config_path or DEFAULT_AGENT_CONFIG_PATH
    all_agents = load_config(path)
    return dict(all_agents.get(agent_name, {}))


async def _run_ask_eval(config: dict[str, Any]) -> dict[str, Any]:
    """Internal async implementation for ask evaluation."""
    agents_host = config.get("agents_host", "https://api.agents.weaviate.io")
    use_async = config.get("use_async", True)
    
    # Get query data
    dataset_name = config.get("ask_dataset")
    docs_collection = config.get("docs_collection")
    queries_input = config.get("queries")
    
    # Load queries
    if queries_input:
        if isinstance(queries_input, AskQueriesCollection):
            # Custom Weaviate collection
            queries = load_ask_queries_from_weaviate(
                collection_name=queries_input.collection_name,
                query_content_key=queries_input.query_content_key,
                answer_key=queries_input.answer_key,
                oracle_context_id_key=queries_input.oracle_context_id_key,
            )
        elif isinstance(queries_input, list):
            # Direct list of InMemoryAskQuery objects
            if not queries_input:
                raise ValueError("Queries list cannot be empty")
            if not all(isinstance(q, InMemoryAskQuery) for q in queries_input):
                raise ValueError(
                    "All queries must be InMemoryAskQuery objects. "
                    f"Found: {set(type(q).__name__ for q in queries_input)}"
                )
            queries = queries_input
        elif isinstance(queries_input, str):
            # Built-in dataset name - loader handles all key mappings internally
            if queries_input in supported_ask_datasets:
                queries = in_memory_ask_dataset_loader(
                    queries_input,
                    longmemeval_subset=config.get("longmemeval_subset"),
                )
            else:
                raise ValueError(
                    f"Unknown ask dataset: '{queries_input}'. "
                    f"Supported datasets: {supported_ask_datasets}. "
                    f"For custom datasets, use AskQueriesCollection or list[InMemoryAskQuery]."
                )
        else:
            raise ValueError(
                f"Queries must be a built-in dataset name (str), AskQueriesCollection, or list[InMemoryAskQuery]. "
                f"Got: {type(queries_input)}"
            )
    else:
        raise ValueError("Must provide 'queries' for ask evaluation")
    
    # Determine dataset identifier
    if docs_collection:
        dataset_identifier = docs_collection.collection_name
    elif dataset_name:
        dataset_identifier = dataset_name
    else:
        dataset_identifier = "custom"
    
    config["dataset_identifier"] = dataset_identifier

    print(f"There are \033[92m{len(queries)}\033[0m total ask queries.\n")
    print("\033[92mFirst Query\033[0m")
    print(f"  Question: {queries[0].question[:100]}...")
    print(f"  Ground Truth Answer: {queries[0].ground_truth_answer[:100]}...")
    if queries[0].oracle_context_id:
        print(f"  Oracle Context ID: {queries[0].oracle_context_id}")

    # Handle subset if requested
    if config.get("use_subset", False):
        import random
        random.seed(config.get("random_seed", 24))
        random.shuffle(queries)
        queries = queries[:config["num_samples"]]
        print(f"Using a subset of {config['num_samples']} queries.")

    # Determine which metric to use
    # MultiHop-RAG uses exact match, OfficeQA uses fuzzy match, others use LLM judge
    use_exact_match = config.get("use_exact_match", False)
    use_officeqa_metric = config.get("use_officeqa_metric", False)
    officeqa_tolerance = config.get("officeqa_tolerance", 0.00)
    if isinstance(queries_input, str) and queries_input == "multihoprag":
        use_exact_match = True
    if isinstance(queries_input, str) and queries_input == "officeqa":
        use_officeqa_metric = True

    # Set system prompt from config override, or fall back to registry
    system_prompt = config.get("system_prompt")
    if not system_prompt and isinstance(queries_input, str):
        system_prompt = get_system_prompt(queries_input)

    # Build agent
    embedding_model = config.get("embedding_model")
    agent_name = config.get("ask_agent_name", "query-agent-ask")
    external_service_host = config.get("external_service_host")

    # Add agent_name to config for result serialization
    config["agent_name"] = agent_name

    if agent_name == "engram":
        use_async = False  # EngramDSPyAgent is sync-only
        agent_cfg = _load_agent_config(agent_name)
        ask_agent = EngramDSPyAgent(
            engram_base_url=agent_cfg.get("engram_base_url", "https://dev-engram.labs.weaviate.io"),
            engram_api_key=agent_cfg.get("engram_api_key"),
            dspy_lm_model=agent_cfg.get("dspy_lm_model", "openai/gpt-5.4"),
            retrieval_limit=agent_cfg.get("retrieval_limit", 10),
            retrieval_type=agent_cfg.get("retrieval_type", "hybrid"),
            engram_group=agent_cfg.get("engram_group", "default"),
            user_id_prefix=agent_cfg.get("user_id_prefix", "longmemeval-"),
        )
    elif docs_collection:
        agent_cfg = _load_agent_config(agent_name)
        ask_agent = AskAgentBuilder(
            agent_name=agent_name,
            docs_collection=docs_collection,
            agents_host=agent_cfg.get("agents_host", agents_host),
            use_async=use_async,
            embedding_model=embedding_model,
            external_service_host=agent_cfg.get("external_service_host", external_service_host),
            system_prompt=system_prompt,
        )
    elif dataset_name:
        agent_cfg = _load_agent_config(agent_name)
        ask_agent = AskAgentBuilder(
            agent_name=agent_name,
            dataset_name=dataset_name,
            agents_host=agent_cfg.get("agents_host", agents_host),
            use_async=use_async,
            embedding_model=embedding_model,
            external_service_host=agent_cfg.get("external_service_host", external_service_host),
            system_prompt=system_prompt,
        )
    else:
        raise ValueError("Must provide 'ask_dataset' or 'docs_collection' for agent initialization")

    # LLM Judge configuration (only used if not using exact match)
    judge_model = config.get("judge_model", "openai/gpt-4.1")
    ensemble_k = config.get("ensemble_k", 3)
    
    num_trials = config.get("num_trials", 1)
    metrics_across_trials = []

    # Run trials
    for trial in range(num_trials):
        print(f"\033[92mRunning ask trial {trial+1}/{num_trials}\033[0m")

        if use_async:        
            print("\033[92mRunning ask queries async!\033[0m")
            await ask_agent.initialize_async()
            
            try:
                results = await run_ask_queries_async(
                    queries=queries,
                    ask_agent=ask_agent,
                    batch_size=config.get("batch_size", 10),
                    max_concurrent=config.get("max_concurrent", 5),
                )
            finally:
                await ask_agent.close_async()
        else:
            print("\n\033[94mRunning synchronous ask benchmark\033[0m")
            results = run_ask_queries(
                queries=queries,
                ask_agent=ask_agent,
                sleep_between_requests=config.get("sleep_between_requests", 0.0),
            )

        # Analyze results with appropriate metric
        metrics = await analyze_ask_results(
            results=results,
            judge_model=judge_model,
            ensemble_k=ensemble_k,
            use_exact_match=use_exact_match,
            use_officeqa_metric=use_officeqa_metric,
            officeqa_tolerance=officeqa_tolerance,
        )
        
        print(f"\n\033[92mTrial {trial+1} Results:\033[0m")
        if use_officeqa_metric:
            score_key = "avg_officeqa_accuracy"
        elif use_exact_match:
            score_key = "avg_exact_match_accuracy"
        else:
            score_key = "avg_alignment_score"
        print(f"  {score_key}: {metrics[score_key]:.2%}")
        print(f"  Avg Query Time: {metrics['avg_query_time']:.2f}s")

        # Extract per-query scores for trial results
        if use_officeqa_metric:
            scores_key = "officeqa_accuracy_scores"
        elif use_exact_match:
            scores_key = "exact_match_accuracy_scores"
        else:
            scores_key = "alignment_score_scores"
        alignment_scores = metrics.get(scores_key, [])

        save_ask_trial_results(
            results=results,
            config=config,
            trial_number=trial+1,
            alignment_scores=alignment_scores,
        )

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


def run_ask_eval(
    queries: Optional[Union[AskQueriesCollection, list[InMemoryAskQuery], str]] = None,
    config_path: Optional[str] = None,
    ask_dataset: Optional[str] = None,
    docs_collection: Optional[DocsCollection] = None,
    agent_name: Optional[str] = None,
    judge_model: Optional[str] = None,
    ensemble_k: Optional[int] = None,
    num_trials: Optional[int] = None,
    use_subset: Optional[bool] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_concurrent: Optional[int] = None,
    use_async: Optional[bool] = None,
    agents_host: Optional[str] = None,
    external_service_host: Optional[str] = None,
    sleep_between_requests: Optional[float] = None,
    output_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    embedding_model: Optional[str] = None,
    longmemeval_subset: Optional[dict] = None,
    **kwargs
) -> dict[str, Any]:
    """
    Run an ask benchmark evaluation.
    
    Evaluates question answering performance using LLM-as-judge for semantic alignment.
    All parameters can be read from benchmark-config.yml if not provided.
    
    Args:
        queries: Built-in dataset name (e.g., "irpapers-visual-queries"),
                 AskQueriesCollection for custom Weaviate collections,
                 or list[InMemoryAskQuery] for direct query objects.
        config_path: Path to YAML config file. Defaults to built-in config.
        ask_dataset: Name of built-in dataset for agent initialization (e.g., "irpapers/images").
        docs_collection: DocsCollection for custom datasets.
        agent_name: Agent to use ("query-agent-ask" or "external").
        judge_model: LLM model for the judge (e.g., "openai/gpt-4.1").
        ensemble_k: Number of ensemble votes for judge.
        num_trials: Number of evaluation trials to run.
        use_subset: Whether to use a random subset of queries.
        num_samples: Number of samples if use_subset is True.
        batch_size: Batch size for async processing.
        max_concurrent: Max concurrent requests for async.
        use_async: Whether to use async execution.
        agents_host: Host URL for the agents service.
        external_service_host: Host URL for external mode (e.g., "http://localhost:8000").
        sleep_between_requests: Seconds to sleep between requests (sync mode only, for rate limiting).
        output_path: Path to save results.
        random_seed: Random seed for reproducibility.
        embedding_model: Embedding model to use.
        **kwargs: Additional config overrides.
        
    Returns:
        Dict containing aggregated metrics across trials.
        
    Example:
        >>> from query_agent_benchmarking import run_ask_eval, InMemoryAskQuery
        >>> 
        >>> # Using config file (simplest)
        >>> results = run_ask_eval()
        >>> 
        >>> # Using built-in dataset
        >>> results = run_ask_eval(
        ...     queries="irpapers-visual-queries",
        ...     ask_dataset="irpapers/images",
        ... )
        >>> 
        >>> # Using custom queries
        >>> queries = [
        ...     InMemoryAskQuery(question="What is HyDE?", ground_truth_answer="..."),
        ... ]
        >>> results = run_ask_eval(queries=queries, docs_collection=my_collection)
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Load base config from file
    file_config = load_config(config_path)
    
    # Build override config from parameters
    override_config = {
        "queries": queries,
        "ask_dataset": ask_dataset,
        "docs_collection": docs_collection,
        "agent_name": agent_name,
        "judge_model": judge_model,
        "ensemble_k": ensemble_k,
        "num_trials": num_trials,
        "use_subset": use_subset,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
        "use_async": use_async,
        "agents_host": agents_host,
        "external_service_host": external_service_host,
        "sleep_between_requests": sleep_between_requests,
        "output_path": output_path,
        "random_seed": random_seed,
        "embedding_model": embedding_model,
        "longmemeval_subset": longmemeval_subset,
        **kwargs
    }
    
    # Merge configs
    final_config = merge_configs(file_config, override_config)
    
    # Run evaluation
    return asyncio.run(_run_ask_eval(final_config))

