"""
Search benchmark service — evaluates ranked retrieval performance.

Orchestrates search mode benchmarks using IR metrics like Recall@K, nDCG@K, etc.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Union

from query_agent_benchmarking.internal.adapters.agents.factory import create_search_agent
from query_agent_benchmarking.internal.adapters.agents.weaviate_search import parse_agent_name
from query_agent_benchmarking.internal.core.ports.search_agent import SearchAgent
from query_agent_benchmarking.internal.adapters.dataset import (
    in_memory_dataset_loader,
    load_queries_from_weaviate_collection,
)
from query_agent_benchmarking.internal.core.domain.models import (
    DocsCollection,
    QueriesCollection,
    InMemoryQuery,
    InMemorySearchQuery,
)
from query_agent_benchmarking.internal.core.domain.query_execution import (
    run_search_queries,
    run_search_queries_async,
    truncate_long_queries,
)
from query_agent_benchmarking.internal.core.domain.analysis import aggregate_metrics
from query_agent_benchmarking.internal.adapters.metrics.ir_metrics_calculator import IRMetricsCalculator
from query_agent_benchmarking.internal.adapters.results.json_file_repository import JsonFileResultRepository
from query_agent_benchmarking.internal.core.domain.display import (
    pretty_print_in_memory_query,
    print_results_comparison,
    print_suite_results,
    print_effort_comparison,
    print_effort_suite,
)
from query_agent_benchmarking.internal.config.loader import load_config, merge_configs
from query_agent_benchmarking.internal.config.config import (
    supported_search_datasets,
    resolve_named_vector_target,
    resolve_embedding_providers,
    get_named_vector_target_entry,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "benchmark-config.yml"
DEFAULT_AGENT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "agent-config.yml"

# Effort levels swept (in order) when ``effort_sweep`` is enabled.
EFFORT_LEVELS = ["low", "medium", "high"]


def _load_agent_config(agent_name: str, agent_config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load agent-specific parameters from agent-config.yml."""
    path = agent_config_path or DEFAULT_AGENT_CONFIG_PATH
    all_agents = load_config(path)
    return dict(all_agents.get(agent_name, {}))


def _baseline_label(agent_name: str) -> str:
    """Display label for a baseline agent: ``hybrid-search[...]`` -> ``hybrid``."""
    base_name, _ = parse_agent_name(agent_name)
    return base_name[: -len("-search")] if base_name.endswith("-search") else base_name


async def _run_effort_sweep(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Run one search eval per effort level and collect results for comparison.

    Runs each level sequentially (so timings are clean and the agents server
    isn't hit by three concurrent sweeps), giving every level its own result
    files via an effort-tagged run id. Baseline agents from
    ``effort_sweep_baselines`` (e.g. "hybrid-search") each run once alongside
    the effort levels as reference points. Per-run config overrides from
    ``effort_sweep_overrides`` (keyed by effort level, baseline agent name, or
    baseline label) are applied on top of the shared config — e.g. to give
    cheap runs higher ``max_concurrent`` than expensive ones. Returns a dict
    mapping each effort level / baseline label to
    ``{"metrics": <aggregated>, "seconds": float, "error": str | None}``.
    """
    sweep_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output = config.get("output_path")
    per_run_overrides = config.get("effort_sweep_overrides") or {}
    results_by_effort: dict[str, dict[str, Any]] = {}

    async def run_tagged(key: str, overrides: dict[str, Any]) -> None:
        # Fresh config per run: _run_search_eval mutates the dict it's given.
        cfg = dict(config)
        cfg.pop("effort_sweep", None)
        cfg.pop("effort_sweep_baselines", None)
        cfg.pop("effort_sweep_overrides", None)
        cfg["sweep_id"] = sweep_id
        cfg["run_id"] = f"{sweep_id}-effort_{key}"
        cfg.update(overrides)
        if base_output:
            path = Path(base_output)
            cfg["output_path"] = str(path.parent / f"{path.stem}_effort_{key}{path.suffix}")

        start = time.perf_counter()
        try:
            aggregated = await _run_search_eval(cfg)
            error = None
        except Exception as exc:  # keep sweeping so one run can't sink the rest
            aggregated = {}
            error = str(exc)
            print(f"\033[91m{key!r} failed: {exc}\033[0m")

        results_by_effort[key] = {
            "metrics": aggregated,
            "seconds": time.perf_counter() - start,
            "error": error,
        }

    for level in EFFORT_LEVELS:
        print(f"\n{'#' * 72}")
        print(f"# Running search benchmark with \033[1meffort={level!r}\033[0m")
        print(f"{'#' * 72}")
        await run_tagged(level, {"effort": level, **per_run_overrides.get(level, {})})

    for agent_name in config.get("effort_sweep_baselines") or []:
        label = _baseline_label(agent_name)
        print(f"\n{'#' * 72}")
        print(f"# Running search benchmark with baseline \033[1m{agent_name!r}\033[0m")
        print(f"{'#' * 72}")
        # The label doubles as the persisted "effort" tag so results files and
        # the console place the baseline alongside the effort levels; non-query-
        # agent adapters ignore the effort parameter at construction time.
        # Overrides may be keyed by the full agent name or its display label.
        await run_tagged(
            label,
            {
                "search_agent_name": agent_name,
                "agent_name": agent_name,
                "effort": label,
                **per_run_overrides.get(agent_name, per_run_overrides.get(label, {})),
            },
        )

    return results_by_effort


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
            "Use compare_search_agents for multiple agents."
        )

    target_vector = config.get("search_target")
    if target_vector is None:
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
        base_name, inline_target = parse_agent_name(raw_agent_name)
        resolved_inline_target = resolve_named_vector_target(
            dataset_name=dataset_identifier,
            target_vector=inline_target,
        )
        return f"{base_name}[{resolved_inline_target}]"

    if target_vector:
        if dataset_identifier and get_named_vector_target_entry(dataset_identifier) is None:
            return raw_agent_name
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

    dataset_name = config.get("search_dataset")
    docs_collection = config.get("docs_collection")
    queries_input = config.get("queries")

    _COLLECTION_OVERRIDES = {
        "reasonir-biology-subset": "BrightBiology_Default",
    }

    if dataset_name:
        if dataset_name not in supported_search_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not supported. "
                f"Supported datasets are: {supported_search_datasets}"
            )

        _, queries = in_memory_dataset_loader(dataset_name, queries_only=True)
        dataset_identifier = dataset_name

        if dataset_name in _COLLECTION_OVERRIDES:
            docs_collection = DocsCollection(
                collection_name=_COLLECTION_OVERRIDES[dataset_name],
                content_key="content",
                id_key="dataset_id",
            )
            dataset_name = None

    elif docs_collection and queries_input:
        if not isinstance(docs_collection, DocsCollection):
            raise ValueError("docs_collection must be a DocsCollection object")

        if isinstance(queries_input, QueriesCollection):
            queries = load_queries_from_weaviate_collection(
                collection_name=queries_input.collection_name,
                query_content_key=queries_input.query_content_key,
                gold_ids_key=queries_input.gold_ids_key,
            )
        elif isinstance(queries_input, list):
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

    if config.get("use_subset", False):
        import random
        random.seed(config.get("random_seed", 24))
        random.shuffle(queries)
        queries = queries[:config["num_samples"]]
        print(f"Using a subset of {config['num_samples']} queries.")

    max_query_tokens = config.get("max_query_tokens")
    if max_query_tokens:
        queries = truncate_long_queries(queries, max_query_tokens)

    # Use a user-provided search agent if available; otherwise build one from config.
    user_provided_agent: Optional[SearchAgent] = config.pop("search_agent", None)

    if user_provided_agent is not None:
        if not isinstance(user_provided_agent, SearchAgent):
            raise TypeError(
                "search_agent must implement the SearchAgent protocol "
                "(run, run_async, initialize_async, close_async). "
                "See query_agent_benchmarking.SearchAgent for the interface."
            )
        query_agent = user_provided_agent
        agent_name = config.get("search_agent_name") or config.get("agent_name") or type(user_provided_agent).__name__
        config["search_agent_name"] = agent_name
        config["agent_name"] = agent_name
    else:
        text_embedding_model, image_embedding_model = _resolve_embedding_models(config)
        agent_name = _resolve_search_agent_name(
            config,
            dataset_identifier=dataset_identifier,
        )
        _, resolved_target_vector = parse_agent_name(agent_name)
        embedding_providers = resolve_embedding_providers(
            dataset_name=dataset_identifier,
            target_vector=resolved_target_vector,
            embedding_providers=config.get("embedding_providers") or config.get("external_providers"),
            embedding_models=[
                text_embedding_model,
                image_embedding_model,
                config.get("embedding_model"),
            ],
        )

        config["search_agent_name"] = agent_name
        config["agent_name"] = agent_name
        if text_embedding_model is not None:
            config["text_embedding_model"] = text_embedding_model
            config["embedding_model"] = text_embedding_model
        if image_embedding_model is not None:
            config["image_embedding_model"] = image_embedding_model
        if embedding_providers:
            config["embedding_providers"] = embedding_providers

        agent_cfg = _load_agent_config(agent_name)
        resolved_agents_host = agent_cfg.get("agents_host", agents_host)
        resolved_external_service_host = agent_cfg.get("external_service_host", config.get("external_service_host"))
        # Query Agent search filtering strategy ("recall" | "precision").
        # Eval-level config takes precedence, then the agent default, then "recall".
        resolved_filtering = config.get("filtering") or agent_cfg.get("filtering") or "recall"
        config["filtering"] = resolved_filtering
        # Query Agent search compute effort ("low" | "medium" | "high"). Eval-level
        # config takes precedence, then the agent default; when unset the agents
        # server applies its own default.
        resolved_effort = config.get("effort") or agent_cfg.get("effort")
        config["effort"] = resolved_effort

        query_agent = create_search_agent(
            agent_name,
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            agents_host=resolved_agents_host,
            embedding_model=text_embedding_model,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            embedding_providers=embedding_providers,
            external_service_host=resolved_external_service_host,
            filtering=resolved_filtering,
            effort=resolved_effort,
        )

    num_trials = config.get("num_trials", 1)
    metrics_across_trials = []

    metrics_calculator = IRMetricsCalculator(
        dataset_name=dataset_identifier,
        extra_metrics=config.get("extra_metrics"),
    )
    result_repo = JsonFileResultRepository()

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
                    max_concurrent=config.get("max_concurrent", 5),
                    sleep_between_requests=config.get("sleep_between_requests", 0),
                )
            finally:
                await query_agent.close_async()
        else:
            print("\n\033[94mRunning synchronous benchmark\033[0m")
            results = run_search_queries(
                queries=queries,
                query_agent=query_agent,
                sleep_between_requests=config.get("sleep_between_requests", 0),
            )

        result_repo.save_trial_results(
            results=results,
            config=config,
            trial_number=trial+1,
        )

        metrics = metrics_calculator.compute(results, queries)
        print(metrics)
        result_repo.save_trial_metrics(
            metrics=metrics,
            config=config,
            trial_number=trial+1,
        )
        metrics_across_trials.append(metrics)

    aggregated_metrics = aggregate_metrics(metrics_across_trials)
    result_repo.save_aggregated_results(
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
    search_agent: Optional[SearchAgent] = None,
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
    filtering: Optional[str] = None,
    effort: Optional[str] = None,
    effort_sweep: Optional[bool] = None,
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
        agent_name: Agent to use ("query-agent-search-mode", "hybrid-search", or "external_service").
            Ignored when search_agent is provided.
        search_agent: A user-provided retriever instance implementing the SearchAgent protocol.
            When provided, the agent_name factory lookup is skipped and this object is used
            directly. Must implement run() and run_async() returning list[ObjectID].
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
        filtering: Query Agent search filtering strategy: "recall" (default —
            generate multiple Weaviate queries spanning different filters and
            interpretations) or "precision" (generate a single query targeting the
            most likely interpretation). Only applies to "query-agent-search-mode".
        effort: Query Agent search compute effort: "low", "medium", or "high".
            Controls how much compute search mode spends per query. When None,
            the agents server applies its own default. Only applies to
            "query-agent-search-mode".
        effort_sweep: When True, run the benchmark once per effort level
            ("low", "medium", "high") and print a side-by-side comparison. Any
            single ``effort`` value is ignored in this mode. Only meaningful for
            "query-agent-search-mode". Baseline agents listed in the
            ``effort_sweep_baselines`` config (e.g. "hybrid-search") each run
            once per sweep and appear alongside the effort levels.
        **kwargs: Additional config overrides.

    Returns:
        Dict containing aggregated metrics across trials. When ``effort_sweep``
        is True, instead returns a dict mapping each effort level to
        ``{"metrics": <aggregated>, "seconds": float, "error": str | None}``.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    file_config = load_config(config_path)

    override_config = {
        "search_dataset": search_dataset,
        "docs_collection": docs_collection,
        "queries": queries,
        "agent_name": agent_name,
        "search_agent": search_agent,
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
        "filtering": filtering,
        "effort": effort,
        "effort_sweep": effort_sweep,
        **kwargs
    }

    final_config = merge_configs(file_config, override_config)

    # An effort sweep only makes sense for the Query Agent search mode; a
    # user-provided search_agent ignores effort, so don't run it three times.
    if final_config.get("effort_sweep") and not final_config.get("search_agent"):
        results_by_effort = asyncio.run(_run_effort_sweep(final_config))
        print_effort_comparison(
            results_by_effort, dataset=final_config.get("search_dataset")
        )
        return results_by_effort

    return asyncio.run(_run_search_eval(final_config))


def compare_search_agents(
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
    filtering: Optional[str] = None,
    effort: Optional[str] = None,
    **kwargs
) -> dict[str, dict[str, Any]]:
    """Run search benchmark for multiple query agents and compare results."""

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    file_config = load_config(config_path)

    agents = agent_names or file_config.get("search_agent_name") or file_config.get("agent_name")
    if agents is None:
        raise ValueError("No agent_names provided. Must specify via parameter or in config file (search_agent_name).")

    agents = [agents] if isinstance(agents, str) else agents

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
        "filtering": filtering,
        "effort": effort,
        **kwargs
    }

    all_results = {}
    for agent in agents:
        agent_override = override_config.copy()
        agent_override["agent_name"] = agent

        if output_path:
            path = Path(output_path)
            agent_override["output_path"] = str(path.parent / f"{path.stem}_{agent}{path.suffix}")

        try:
            all_results[agent] = asyncio.run(_run_search_eval(merge_configs(file_config, agent_override)))
        except Exception as e:
            all_results[agent] = {"error": str(e)}

    print_results_comparison(all_results)

    return all_results


def run_search_evals(
    config_path: Optional[str] = None,
    search_datasets: Optional[list[str]] = None,
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
    filtering: Optional[str] = None,
    effort: Optional[str] = None,
    effort_sweep: Optional[bool] = None,
    **kwargs
) -> dict[str, dict[str, Any]]:
    """
    Run search benchmark across multiple datasets and compare results.

    Iterates over a list of search datasets, running a full evaluation for each,
    then prints a cross-dataset comparison summary.

    Args:
        config_path: Path to YAML config file. Defaults to built-in config.
        search_datasets: List of dataset names (e.g., ["bright/biology", "bright/economics"]).
            Falls back to ``search_datasets`` in the config file.
        agent_name: Agent to use for all datasets.
        num_trials: Number of evaluation trials per dataset.
        use_subset: Whether to use a random subset of queries.
        num_samples: Number of samples if use_subset is True.
        batch_size: Batch size for async processing.
        max_concurrent: Max concurrent requests for async.
        use_async: Whether to use async execution.
        agents_host: Host URL for the agents service.
        external_service_host: Host URL for external mode.
        output_path: Base path to save results (dataset name is appended).
        random_seed: Random seed for reproducibility.
        embedding_model: Deprecated alias for text_embedding_model.
        text_embedding_model: Text embedding model.
        image_embedding_model: Image/multimodal embedding model.
        embedding_providers: Provider list for header injection.
        search_target: Canonical named vector(s).
        search_target_vector: Legacy vector-name key.
        filtering: Query Agent search filtering strategy, "recall" (default) or
            "precision". Only applies to "query-agent-search-mode".
        effort: Query Agent search compute effort, "low", "medium", or "high".
            When None, the agents server applies its own default. Only applies
            to "query-agent-search-mode".
        effort_sweep: When True, sweep every effort level ("low", "medium",
            "high") for each dataset and print a per-dataset comparison. Only
            meaningful for "query-agent-search-mode".
        **kwargs: Additional config overrides.

    Returns:
        Dict mapping dataset name to its aggregated metrics. When
        ``effort_sweep`` is True, each dataset instead maps to its
        ``results_by_effort`` dict (effort level -> metrics/seconds/error).
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    file_config = load_config(config_path)

    datasets = search_datasets or file_config.get("search_datasets")
    if not datasets:
        raise ValueError(
            "No search_datasets provided. Specify via parameter or "
            "in config file (search_datasets list)."
        )

    if isinstance(datasets, str):
        datasets = [datasets]

    override_config = {
        "agent_name": agent_name,
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
        "filtering": filtering,
        "effort": effort,
        "effort_sweep": effort_sweep,
        **kwargs
    }

    effort_sweep_on = bool(
        effort_sweep if effort_sweep is not None else file_config.get("effort_sweep")
    )

    suite_results: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"\033[92mDataset: {dataset}\033[0m")
        print(f"{'=' * 60}\n")

        dataset_override = override_config.copy()
        dataset_override["search_dataset"] = dataset

        if output_path:
            path = Path(output_path)
            safe_name = dataset.replace("/", "_")
            dataset_override["output_path"] = str(
                path.parent / f"{path.stem}_{safe_name}{path.suffix}"
            )

        merged = merge_configs(file_config, dataset_override)
        try:
            if effort_sweep_on:
                suite_results[dataset] = asyncio.run(_run_effort_sweep(merged))
            else:
                suite_results[dataset] = asyncio.run(_run_search_eval(merged))
        except Exception as e:
            print(f"\033[91mError running {dataset}: {e}\033[0m")
            suite_results[dataset] = {"error": str(e)}

    if effort_sweep_on:
        print_effort_suite(suite_results)
    else:
        print_suite_results(suite_results)

    return suite_results
