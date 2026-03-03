"""
Core benchmark functions for both search and ask mode evaluations.

This module contains the low-level query execution and analysis functions
used by both search_benchmark_run.py and ask_benchmark_run.py.
"""

import asyncio
import time
from typing import Any, Optional
from tqdm import tqdm
import numpy as np

from query_agent_benchmarking.metrics.ir_metrics import (
    calculate_recall_at_k, 
    calculate_success_at_k,
    calculate_coverage, 
    calculate_alpha_ndcg,
    calculate_nDCG_at_k
)
from query_agent_benchmarking.metrics.lmjudge_alignment import LMJudge
from query_agent_benchmarking.metrics.exact_match import calculate_exact_match
from query_agent_benchmarking.models import (
    QueryResult,
    InMemoryQuery,
    InMemoryAskQuery,
    AskResult,
)


# ============================================================================
# Dataset-to-Metrics Registry
# ============================================================================

_DEFAULT_METRICS = [
    {"func": calculate_recall_at_k, "params": {"k": 1}},
    {"func": calculate_recall_at_k, "params": {"k": 5}},
    {"func": calculate_recall_at_k, "params": {"k": 20}},
    {"func": calculate_nDCG_at_k, "params": {"k": 10}},
]

DATASET_METRICS: dict[str, list[dict]] = {
    "enron": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
    ],
    "wixqa": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
    ],
    "freshstack-": [
        {"func": calculate_recall_at_k, "params": {"k": 50}},
        {"func": calculate_coverage, "params": {"k": 5}},
        {"func": calculate_coverage, "params": {"k": 10}},
        {"func": calculate_coverage, "params": {"k": 20}},
        {"func": calculate_alpha_ndcg, "params": {"alpha": 0.5, "k": 10}},
    ],
    "beir/": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
        {"func": calculate_nDCG_at_k, "params": {"k": 10}},
    ],
    "lotte/": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
        {"func": calculate_success_at_k, "params": {"k": 5}},
    ],
    "bright/": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
        {"func": calculate_nDCG_at_k, "params": {"k": 10}},
    ],
    "irpapers": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
    ],
    "vidore_v3_hr": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
        {"func": calculate_nDCG_at_k, "params": {"k": 10}},
    ],
}


def _resolve_metrics(dataset_name: Optional[str]) -> list[dict]:
    """Resolve the metrics config for a given dataset name using prefix matching."""
    if dataset_name is None:
        return _DEFAULT_METRICS
    for prefix, metrics in DATASET_METRICS.items():
        if dataset_name == prefix or dataset_name.startswith(prefix):
            return metrics
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================================
# Search Mode Functions
# ============================================================================

def run_search_queries(
    queries: list[InMemoryQuery],
    query_agent: Any,
) -> list[QueryResult]:
    """Synchronous version of search query execution."""
    results = []
    start = time.time()
    for i, query in enumerate(tqdm(queries, desc="Running search queries")):
        query_start_time = time.time()
        stringified_ids = [str(dataset_id) for dataset_id in query.dataset_ids]
        response = query_agent.run(query.question)  # -> list[ObjectID]
        query_time_taken = time.time() - query_start_time

        results.append(QueryResult(
            query=query,
            query_ground_truth_id=stringified_ids,
            retrieved_ids=response,
            time_taken=query_time_taken
        ))
        
        if i % 10 == 0:
            print(f"\n\033[93m--- Progress Update ({i}/{len(queries)}) ---\033[0m")
            print(f"Latest query: {query.question}")
            print(f"Ground truth: {query.dataset_ids}")
            print(f"Latest response: {results[i].retrieved_ids}")
            print(f"Time taken: {query_time_taken:.2f} seconds")
            
    print(f"\033[95mExperiment completed {len(results)} queries in {time.time() - start:.2f} seconds.\033[0m")
    return results



async def run_search_queries_async(
    queries: list[InMemoryQuery],
    query_agent: Any,
    batch_size: int = 10,
    max_concurrent: int = 3
) -> list[QueryResult]:
    """
    Asynchronous version of search query execution with concurrent execution.
    
    Args:
        queries: List of search queries
        query_agent: Async search agent
        batch_size: Number of queries to process in each batch
        max_concurrent: Maximum number of concurrent requests
    """
    results = []
    start = time.time()
    
    # Limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_query(query, index, retry_count=0, max_retries=3):
        async with semaphore:
            query_start_time = time.time()
            stringified_ids = [str(dataset_id) for dataset_id in query.dataset_ids]
            try:
                if retry_count > 0:
                    delay = min(2 ** retry_count, 10)  # Exponential backoff
                    print(f"\nRetrying query {index} (attempt {retry_count + 1}) after {delay}s delay...")
                    await asyncio.sleep(delay)
                elif index > 0:
                    await asyncio.sleep(0.1)
                
                question_sample = query.question
                print(f"Running search query {index}: {question_sample}")
                response = await query_agent.run_async(query.question)
                query_time_taken = time.time() - query_start_time

                result = QueryResult(
                    query=query,
                    query_ground_truth_id=stringified_ids,
                    retrieved_ids=response,
                    time_taken=query_time_taken
                )

                return result
            except Exception as e:
                error_msg = str(e)
                query_time_taken = time.time() - query_start_time
                
                print(f"\n\033[91mError processing query {index}: {error_msg}\033[0m")
                return QueryResult(
                    query=query,
                    query_ground_truth_id=stringified_ids,
                    retrieved_ids=[],
                    time_taken=query_time_taken
                )
    
    queries_to_process = queries
    total_batches = (len(queries_to_process) + batch_size - 1) // batch_size
    
    print(f"\033[94mProcessing {len(queries_to_process)} search queries in {total_batches} batches "
          f"(batch_size={batch_size}, max_concurrent={max_concurrent})\033[0m")
    
    for batch_idx in range(0, len(queries_to_process), batch_size):
        batch = queries_to_process[batch_idx:batch_idx + batch_size]
        batch_start = time.time()
        
        print(f"\nStarting batch {batch_idx // batch_size + 1}/{total_batches}")
        
        tasks = [
            process_query(query, batch_idx + i) 
            for i, query in enumerate(batch)
        ]
        
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        batch_time = time.time() - batch_start
        completed = min(batch_idx + batch_size, len(queries_to_process))
        
        batch_successes = sum(1 for r in batch_results if "error" not in r)
        batch_errors = len(batch_results) - batch_successes
        
        print(f"\n\033[93m--- Batch {batch_idx // batch_size + 1} Complete ({completed}/{len(queries_to_process)}) ---\033[0m")
        print(f"Successes: {batch_successes}, Errors: {batch_errors}")
        print(f"Batch completed in {batch_time:.2f} seconds")
        
        successful_results = [r for r in batch_results if "error" not in r]
        if successful_results:
            sample = successful_results[0]
            print(f"Sample query: {sample.query.question[:100]}...")
            print(f"Sample response: {sample.retrieved_ids[:200]}...")
        
        if batch_idx + batch_size < len(queries_to_process):
            await asyncio.sleep(1)
    
    total_time = time.time() - start
    total_successes = sum(1 for r in results if "error" not in r)
    total_errors = len(results) - total_successes
    
    print("\n\033[95mAsync search experiment completed!\033[0m")
    print(f"\033[95mResults: {total_successes} successful, {total_errors} failed out of {len(results)} total\033[0m")
    print(f"\033[95mTotal time: {total_time:.2f} seconds\033[0m")
    print(f"\033[95mAverage time per query: {total_time/len(results):.2f} seconds\033[0m")
    
    return results




# ============================================================================
# Ask Mode Functions
# ============================================================================

def run_ask_queries(
    queries: list[InMemoryAskQuery],
    ask_agent: Any,
    sleep_between_requests: float = 0.0,
) -> list[AskResult]:
    """Synchronous version of ask query execution."""
    results = []
    start = time.time()
    
    for i, query in enumerate(tqdm(queries, desc="Running ask queries")):
        query_start_time = time.time()
        
        try:
            # Pass oracle_context_id if available (for external mode)
            response = ask_agent.run(
                query=query.question,
                oracle_context_id=query.oracle_context_id
            )
            query_time_taken = time.time() - query_start_time
            
            results.append(AskResult(
                query=query,
                system_answer=response.final_answer,
                time_taken=query_time_taken,
            ))
        except Exception as e:
            query_time_taken = time.time() - query_start_time
            print(f"\n\033[91mError processing ask query {i}: {str(e)}\033[0m")
            results.append(AskResult(
                query=query,
                system_answer=f"[ERROR] {str(e)}",
                time_taken=query_time_taken,
            ))
        
        if i % 10 == 0:
            print(f"\n\033[93m--- Progress Update ({i}/{len(queries)}) ---\033[0m")
            print(f"Latest query: {query.question[:100]}...")
            print(f"Latest answer: {results[i].system_answer[:200]}...")
            print(f"Time taken: {query_time_taken:.2f} seconds")
        
        # Sleep between requests for rate limiting
        if sleep_between_requests > 0 and i < len(queries) - 1:
            time.sleep(sleep_between_requests)
            
    print(f"\033[95mAsk experiment completed {len(results)} queries in {time.time() - start:.2f} seconds.\033[0m")
    return results


async def run_ask_queries_async(
    queries: list[InMemoryAskQuery],
    ask_agent: Any,
    batch_size: int = 10,
    max_concurrent: int = 3,
) -> list[AskResult]:
    """
    Asynchronous version of ask query execution with concurrent execution.
    
    Args:
        queries: List of ask queries
        ask_agent: Async ask agent
        batch_size: Number of queries to process in each batch
        max_concurrent: Maximum number of concurrent requests
    """
    results = []
    start = time.time()
    
    # Limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_query(query: InMemoryAskQuery, index: int):
        async with semaphore:
            query_start_time = time.time()
            try:
                if index > 0:
                    await asyncio.sleep(0.1)
                
                print(f"Running ask query {index}: {query.question[:50]}...")
                response = await ask_agent.run_async(
                    query=query.question,
                    oracle_context_id=query.oracle_context_id
                )
                query_time_taken = time.time() - query_start_time

                return AskResult(
                    query=query,
                    system_answer=response.final_answer,
                    time_taken=query_time_taken,
                )
            except Exception as e:
                error_msg = str(e)
                query_time_taken = time.time() - query_start_time
                
                print(f"\n\033[91mError processing ask query {index}: {error_msg}\033[0m")
                return AskResult(
                    query=query,
                    system_answer=f"[ERROR] {error_msg}",
                    time_taken=query_time_taken,
                )
    
    total_batches = (len(queries) + batch_size - 1) // batch_size
    
    print(f"\033[94mProcessing {len(queries)} ask queries in {total_batches} batches "
          f"(batch_size={batch_size}, max_concurrent={max_concurrent})\033[0m")
    
    for batch_idx in range(0, len(queries), batch_size):
        batch = queries[batch_idx:batch_idx + batch_size]
        batch_start = time.time()
        
        print(f"\nStarting batch {batch_idx // batch_size + 1}/{total_batches}")
        
        tasks = [
            process_query(query, batch_idx + i) 
            for i, query in enumerate(batch)
        ]
        
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        batch_time = time.time() - batch_start
        completed = min(batch_idx + batch_size, len(queries))
        
        batch_errors = sum(1 for r in batch_results if r.system_answer.startswith("[ERROR]"))
        batch_successes = len(batch_results) - batch_errors
        
        print(f"\n\033[93m--- Batch {batch_idx // batch_size + 1} Complete ({completed}/{len(queries)}) ---\033[0m")
        print(f"Successes: {batch_successes}, Errors: {batch_errors}")
        print(f"Batch completed in {batch_time:.2f} seconds")
        
        if batch_idx + batch_size < len(queries):
            await asyncio.sleep(1)
    
    total_time = time.time() - start
    total_errors = sum(1 for r in results if r.system_answer.startswith("[ERROR]"))
    total_successes = len(results) - total_errors
    
    print("\n\033[95mAsync ask experiment completed!\033[0m")
    print(f"\033[95mResults: {total_successes} successful, {total_errors} failed out of {len(results)} total\033[0m")
    print(f"\033[95mTotal time: {total_time:.2f} seconds\033[0m")
    print(f"\033[95mAverage time per query: {total_time/len(results):.2f} seconds\033[0m")
    
    return results

# ============================================================================
# Search Results Analysis
# ============================================================================

async def analyze_search_results(
    results: list[QueryResult],
    ground_truths: list[InMemoryQuery],
    dataset_name: Optional[str] = None,
):
    """Analyze search results with dataset-specific IR metrics."""
    
    # Resolve dataset-specific metrics from the registry
    metrics = _resolve_metrics(dataset_name)
    
    # Initialize result storage with descriptive keys
    metric_results = {}
    for config in metrics:
        name = config["func"].__name__.replace("calculate_", "")
        if "k" in config["params"]:
            key = f"{name}_at_{config['params']['k']}"
        else:
            key = name
        metric_results[key] = []
        
    query_times = []
    
    for i, (result, ground_truth) in enumerate(tqdm(zip(results, ground_truths), desc="Analyzing search results")):
        if result.retrieved_ids == []:  # proxy for error
            print(f"\n\033[91mSkipping analysis for query {i} due to error.\033[0m")
            continue

        # Corrected list comprehension
        retrieved_ids = [res.object_id for res in result.retrieved_ids]
        
        for metric_config in metrics:
            metric_func = metric_config["func"]
            params = metric_config["params"]
            func_name = metric_func.__name__
            
            # Determine the key for storing the result
            key = func_name.replace("calculate_", "")
            if "k" in params:
                key = f"{key}_at_{params['k']}"

            # Call metric function with the correct arguments
            score = 0.0
            if "recall" in func_name:
                score = metric_func(
                    target_ids=ground_truth.dataset_ids,
                    retrieved_ids=retrieved_ids,
                    **params
                )
            elif func_name in ["calculate_coverage", "calculate_alpha_ndcg"]:
                if ground_truth.nugget_data:
                    score = metric_func(
                        retrieved_ids=retrieved_ids, 
                        nugget_data=ground_truth.nugget_data, 
                        **params
                    )
                else:
                    score = 0.0
            elif "nDCG" in func_name or "ndcg" in func_name.lower():
                # Handle nDCG calculation
                score = metric_func(
                    target_ids=ground_truth.dataset_ids,
                    retrieved_ids=retrieved_ids,
                    **params
                )
            else:
                # Fallback for any other metric functions
                # Try calling with the standard signature first
                try:
                    score = metric_func(
                        target_ids=ground_truth.dataset_ids,
                        retrieved_ids=retrieved_ids,
                        **params
                    )
                except TypeError:
                    # If that fails, try without target_ids (for metrics that don't need ground truth)
                    score = metric_func(
                        retrieved_ids=retrieved_ids,
                        **params
                    )
            
            metric_results[key].append(score)
        
        query_times.append(result.time_taken)
        
        if (i + 1) % 10 == 0:
            print(f"\n\033[93m--- Analysis Progress ({i + 1}/{len(results)}) ---\033[0m")
            for metric_name, scores in metric_results.items():
                if scores:
                    display_name = metric_name.replace("_", " ").title()
                    print(f"Current average {display_name}: {np.mean(scores):.2f}")
            
            print(f"Current average query time: {np.mean(query_times):.2f} seconds")
    
    # Build results dictionary
    results_dict = {
        "avg_query_time": np.mean(query_times) if query_times else 0,
        "query_times": query_times,
    }
    
    for metric_name, scores in metric_results.items():
        results_dict[f"avg_{metric_name}"] = np.mean(scores) if scores else 0
        results_dict[f"{metric_name}_scores"] = scores
    
    # Print summary
    print("\n\033[92m===== Search Benchmark Results =====\033[0m")
    print(f"Dataset: {dataset_name}")
    print(f"Number of queries: {len(results)}")
    
    for metric_name, scores in metric_results.items():
        if scores:
            display_name = metric_name.replace("_", " ").title()
            print(f"Average {display_name}: {np.mean(scores):.2f}")
    
    print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")
    
    return results_dict




# ============================================================================
# Ask Results Analysis
# ============================================================================

async def analyze_ask_results(
    results: list[AskResult],
    judge_model: str = "openai/gpt-4.1",
    ensemble_k: int = 3,
    use_exact_match: bool = False,
) -> dict:
    """
    Analyze ask results using LLM-as-judge for semantic alignment or exact match.
    
    Args:
        results: List of AskResult objects from ask query execution.
        judge_model: LLM model for the judge (only used if use_exact_match=False).
        ensemble_k: Number of ensemble votes for judge (only used if use_exact_match=False).
        use_exact_match: If True, use exact string matching instead of LLM judge.
        
    Returns:
        Dictionary with alignment scores and timing metrics.
    """
    if use_exact_match:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with exact match...\033[0m")
    else:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with LLM judge...\033[0m")
        print(f"Judge model: {judge_model}, Ensemble K: {ensemble_k}")
    
    # Initialize the LLM judge only if not using exact match
    judge = None if use_exact_match else LMJudge(model=judge_model, ensemble_k=ensemble_k)
    
    alignment_scores = []
    query_times = []
    misaligned_indices = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    desc = "Running exact match" if use_exact_match else "Running LLM judge"
    for i, result in enumerate(tqdm(results, desc=desc)):
        # Skip error results
        if result.system_answer.startswith("[ERROR]"):
            print(f"\n\033[91mSkipping evaluation for query {i} due to error.\033[0m")
            continue
        
        if use_exact_match:
            # Use exact match
            aligned = calculate_exact_match(
                system_answer=result.system_answer,
                ground_truth_answer=result.query.ground_truth_answer,
            )
            alignment_scores.append(1 if aligned else 0)
        else:
            # Run judge with details
            judge_result = judge.evaluate_with_details(
                question=result.query.question,
                system_answer=result.system_answer,
                correct_answer=result.query.ground_truth_answer,
            )
            
            aligned = judge_result["aligned"]
            alignment_scores.append(1 if aligned else 0)
            total_input_tokens += judge_result.get("input_tokens", 0)
            total_output_tokens += judge_result.get("output_tokens", 0)
        
        query_times.append(result.time_taken)
        
        if not aligned:
            misaligned_indices.append(i)
        
        # Progress update
        if (i + 1) % 5 == 0:
            current_avg = np.mean(alignment_scores) if alignment_scores else 0
            metric_name = "exact match" if use_exact_match else "alignment"
            print(f"\n\033[93m--- Evaluation Progress ({i + 1}/{len(results)}) ---\033[0m")
            print(f"Running {metric_name} score: {current_avg:.2%}")
            print(f"Running avg query time: {np.mean(query_times):.2f}s")
    
    # Build results dictionary (same structure as search benchmark)
    metric_name = "exact_match_accuracy" if use_exact_match else "alignment_score"
    metric_results = {metric_name: alignment_scores}

    results_dict = {
        "avg_query_time": np.mean(query_times) if query_times else 0,
        "query_times": query_times,
        "misaligned_indices": misaligned_indices,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "metric": "exact_match" if use_exact_match else "llm_judge",
    }

    for name, scores in metric_results.items():
        results_dict[f"avg_{name}"] = np.mean(scores) if scores else 0
        results_dict[f"{name}_scores"] = scores

    if not use_exact_match:
        results_dict["judge_model"] = judge_model
        results_dict["ensemble_k"] = ensemble_k

    # Print summary
    print("\n\033[92m===== Ask Benchmark Results =====\033[0m")
    print(f"Number of queries evaluated: {len(alignment_scores)}")
    for name, scores in metric_results.items():
        if scores:
            display_name = name.replace("_", " ").title()
            print(f"Average {display_name}: {np.mean(scores):.2%}")
    print(f"Misaligned queries: {len(misaligned_indices)}")
    print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")
    if not use_exact_match:
        print(f"Total Judge Tokens: {total_input_tokens + total_output_tokens}")

    return results_dict


def aggregate_metrics(metrics_across_trials: list[dict]) -> dict:
    """Aggregate metrics from multiple trials into statistical summaries."""
    
    if not metrics_across_trials:
        return {}
    
    # Get all metric keys that start with "avg_" from first trial
    avg_keys = [k for k in metrics_across_trials[0].keys() if k.startswith("avg_")]
    
    aggregated = {
        "num_trials": len(metrics_across_trials),
        "trials": []  # Store individual trial averages
    }
    
    # Calculate statistics for each metric
    for key in avg_keys:
        values = [trial[key] for trial in metrics_across_trials if key in trial]
        
        if values:
            metric_name = key  # Keep the "avg_" prefix for clarity
            aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))
            aggregated[f"{metric_name}_min"] = float(np.min(values))
            aggregated[f"{metric_name}_max"] = float(np.max(values))
    
    # Store individual trial summaries for reference
    for i, trial in enumerate(metrics_across_trials):
        trial_summary = {
            "trial": i + 1,
            **{k: v for k, v in trial.items() if k.startswith("avg_")}
        }
        aggregated["trials"].append(trial_summary)
    
    # Print summary to console
    print("\n" + "="*70)
    print(f"📊 AGGREGATED RESULTS ({len(metrics_across_trials)} trials)")
    print("="*70)
    
    for key in avg_keys:
        values = [trial[key] for trial in metrics_across_trials if key in trial]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            metric_display = key.replace("_", " ").title()
            
            print(f"\n{metric_display}:")
            print(f"  Mean: {mean:.4f} (± {std:.4f})")
            print(f"  Min:  {min_val:.4f}")
            print(f"  Max:  {max_val:.4f}")
            print(f"  Raw:  {[f'{v:.4f}' for v in values]}")
    
    print("\n" + "="*70 + "\n")
    
    return aggregated