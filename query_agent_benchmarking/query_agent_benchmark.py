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
from query_agent_benchmarking.models import QueryResult, InMemoryQuery

def run_queries(
    queries: list[InMemoryQuery],
    query_agent: Any,
) -> list[QueryResult]:
    """Synchronous version of run_queries"""
    results = []
    start = time.time()
    for i, query in enumerate(tqdm(queries, desc="Running queries")):
        query_start_time = time.time()
        stringified_ids = [str(dataset_id) for dataset_id in query.dataset_ids]
        response = query_agent.run(query.question) # -> list[ObjectID]
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

async def run_queries_async(
    queries: list[InMemoryQuery],
    query_agent: Any,
    batch_size: int = 10,
    max_concurrent: int = 3
) -> list[QueryResult]:
    """
    Asynchronous version of run_queries with concurrent execution.
    
    Args:
        queries: List of query dictionaries
        query_agent: Async searcher
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
                print(f"Running query {index}: {question_sample}")
                response = await query_agent.run_async(query.question)
                query_time_taken = time.time() - query_start_time

                results = QueryResult(
                    query=query,
                    query_ground_truth_id=stringified_ids,
                    retrieved_ids=response,
                    time_taken=query_time_taken
                )

                return results
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
    
    print(f"\033[94mProcessing {len(queries_to_process)} queries in {total_batches} batches "
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
    
    print("\n\033[95mAsync experiment completed!\033[0m")
    print(f"\033[95mResults: {total_successes} successful, {total_errors} failed out of {len(results)} total\033[0m")
    print(f"\033[95mTotal time: {total_time:.2f} seconds\033[0m")
    print(f"\033[95mAverage time per query: {total_time/len(results):.2f} seconds\033[0m")
    
    return results

async def analyze_results(
    results: list[QueryResult],
    ground_truths: list[InMemoryQuery],
    dataset_name: Optional[str] = None,
):
    """Analyze results with dataset-specific metrics."""
    
    # Define metrics with their specific parameters for each dataset
    if dataset_name is None:
        metrics = [
            {"func": calculate_recall_at_k, "params": {"k": 1}},
            {"func": calculate_recall_at_k, "params": {"k": 5}},
            {"func": calculate_recall_at_k, "params": {"k": 20}},
            {"func": calculate_nDCG_at_k, "params": {"k": 10}},
        ]
    else:
        if dataset_name == "enron":
            metrics = [
                {"func": calculate_recall_at_k, "params": {"k": 1}},
                {"func": calculate_recall_at_k, "params": {"k": 5}},
                {"func": calculate_recall_at_k, "params": {"k": 20}},
            ]
        elif dataset_name == "wixqa":
            metrics = [
                {"func": calculate_recall_at_k, "params": {"k": 1}},
                {"func": calculate_recall_at_k, "params": {"k": 5}},
                {"func": calculate_recall_at_k, "params": {"k": 20}},
            ]
        elif dataset_name.startswith("freshstack-"):
            metrics = [
                {"func": calculate_coverage, "params": {"k": 1000}},
                {"func": calculate_alpha_ndcg, "params": {"alpha": 0.5, "k": 10}},
            ]
        elif dataset_name.startswith("beir/"):
            metrics = [
                {"func": calculate_recall_at_k, "params": {"k": 1}},
                {"func": calculate_recall_at_k, "params": {"k": 5}},
                {"func": calculate_recall_at_k, "params": {"k": 20}},
                {"func": calculate_nDCG_at_k, "params": {"k": 10}},
            ]
        elif dataset_name.startswith("lotte/"):
            metrics = [
                {"func": calculate_recall_at_k, "params": {"k": 1}},
                {"func": calculate_recall_at_k, "params": {"k": 5}},
                {"func": calculate_recall_at_k, "params": {"k": 20}},
                {"func": calculate_success_at_k, "params": {"k": 5}},
            ]
        elif dataset_name.startswith("bright/"):
            metrics = [
                {"func": calculate_recall_at_k, "params": {"k": 1}},
                {"func": calculate_recall_at_k, "params": {"k": 5}},
                {"func": calculate_recall_at_k, "params": {"k": 20}},
                {"func": calculate_nDCG_at_k, "params": {"k": 10}},
            ]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Initialize result storage with descriptive keys
    metric_results = {}
    for config in metrics:
        name = config["func"].__name__.replace("calculate_", "")
        if "recall" in name and "k" in config["params"]:
            key = f"recall_at_{config['params']['k']}"
        else:
            key = name
        metric_results[key] = []
        
    query_times = []
    
    for i, (result, ground_truth) in enumerate(tqdm(zip(results, ground_truths))):
        if result.retrieved_ids == []: # proxy for error
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
            if "recall" in key and "k" in params:
                key = f"recall_at_{params['k']}"

            # Call metric function with the correct arguments
            score = 0.0
            if "recall" in func_name:
                score = metric_func(
                    target_ids=ground_truth.dataset_ids,
                    retrieved_ids=retrieved_ids,
                    **params
                )
            elif func_name in ["calculate_coverage", "calculate_alpha_ndcg"]:
                pass
                """
                if ground_truth.get(nugget_data):
                    for idx, nugget in enumerate(ground_truth.nugget_data):
                        if 'id' not in nugget:
                            nugget['id'] = f"nugget_{idx}"
                
                score = metric_func(
                    retrieved_ids=retrieved_ids, 
                    nuggets=ground_truth.nugget_data, 
                    **params
                )
                """
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
    print("\n\033[92m===== Benchmark Results =====\033[0m")
    print(f"Dataset: {dataset_name}")
    print(f"Number of queries: {len(results)}")
    
    for metric_name, scores in metric_results.items():
        if scores:
            display_name = metric_name.replace("_", " ").title()
            print(f"Average {display_name}: {np.mean(scores):.2f}")
    
    print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")
    
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