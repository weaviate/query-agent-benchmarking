import time
import asyncio
from typing import Any
from tqdm import tqdm
import numpy as np
from benchmarker.metrics.ir_metrics import calculate_recall, calculate_coverage, calculate_alpha_ndcg
from benchmarker.models import ObjectID

def run_queries(
    queries: list[dict],
    query_agent: Any,
    num_samples: int
) -> list[dict]:
    """Synchronous version of run_queries"""
    results = []
    start = time.time()
    for i, query in enumerate(tqdm(queries[:num_samples], desc="Running queries")):
        query_start_time = time.time()
        response = query_agent.run(query["question"]) # -> list[ObjectID]
        query_time_taken = time.time() - query_start_time

        results.append({
            "query": query,
            "query_id": query["dataset_ids"],
            "retrieved_ids": response,
            "time_taken": query_time_taken
        })
        
        # Print rolling update every 10 queries
        if (i + 1) % 10 == 0:
            print(f"\n\033[93m--- Progress Update ({i + 1}/{num_samples}) ---\033[0m")
            print(f"Latest query: {query['question']}")
            print(f"Time taken: {query_time_taken:.2f} seconds")
            
    print(f"\033[95mExperiment completed {len(results)} queries in {time.time() - start:.2f} seconds.\033[0m")
    return results

async def run_queries_async(
    queries: list[dict],
    query_agent: Any,
    num_samples: int,
    batch_size: int = 10,
    max_concurrent: int = 3  # Reduced default to avoid rate limiting
):
    pass
    '''
    """
    Asynchronous version of run_queries with concurrent execution.
    
    Args:
        queries: List of query dictionaries
        query_agent: Async agent instance
        num_samples: Number of queries to run
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
            try:
                if retry_count > 0:
                    delay = min(2 ** retry_count, 10)  # Exponential backoff
                    print(f"\nRetrying query {index} (attempt {retry_count + 1}) after {delay}s delay...")
                    await asyncio.sleep(delay)
                elif index > 0:
                    await asyncio.sleep(0.1)
                
                response = await query_agent.run_async(query["question"])
                query_time_taken = time.time() - query_start_time
                
                total_searches = len(response.searches) if response.searches else 0
                total_aggregations = len(response.aggregations) if response.aggregations else 0
                
                return {
                    "query": query,
                    "query_id": query["dataset_ids"],
                    "answer": response.final_answer,
                    "sources": response.sources,
                    "num_searches": total_searches,
                    "num_aggregations": total_aggregations,
                    "misc_response": response,
                    "time_taken": query_time_taken,
                    "index": index  # Keep track of original order
                }
            except Exception as e:
                error_msg = str(e)
                query_time_taken = time.time() - query_start_time
                
                print(f"\n\033[91mError processing query {index}: {error_msg}\033[0m")
                return {
                    "query": query,
                    "query_id": query["dataset_ids"],
                    "answer": "",
                    "sources": [],
                    "num_searches": 0,
                    "num_aggregations": 0,
                    "misc_response": None,
                    "time_taken": query_time_taken,
                    "index": index,
                    "error": error_msg
                }
    
    queries_to_process = queries[:num_samples]
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
            print(f"Sample query: {sample['query']['question'][:100]}...")
            print(f"Sample response: {sample['answer'][:200]}...")
        
        if batch_idx + batch_size < len(queries_to_process):
            await asyncio.sleep(1)
    
    results.sort(key=lambda x: x["index"])
    
    for result in results:
        result.pop("index", None)
    
    total_time = time.time() - start
    total_successes = sum(1 for r in results if "error" not in r)
    total_errors = len(results) - total_successes
    
    print("\n\033[95mAsync experiment completed!\033[0m")
    print(f"\033[95mResults: {total_successes} successful, {total_errors} failed out of {len(results)} total\033[0m")
    print(f"\033[95mTotal time: {total_time:.2f} seconds\033[0m")
    print(f"\033[95mAverage time per query: {total_time/len(results):.2f} seconds\033[0m")
    
    return results
    '''

async def analyze_results(
    weaviate_client: Any,
    dataset_name: str,
    retrieved_ids: list[list[ObjectID]], # Will neded to extend this for testing multi-collection search
    ground_truths: list[dict],
):
    """Analyze results with dataset-specific metrics."""
    
    # Get collection and determine which metrics to use
    if dataset_name == "enron":
        metrics = [calculate_recall]
    elif dataset_name == "wixqa":
        metrics = [calculate_recall]
    elif dataset_name.startswith("freshstack-"):
        metrics = [calculate_recall, calculate_coverage, calculate_alpha_ndcg]
    else:
        raise Exception("Enter a valid dataset_name!")
    
    # Initialize result storage
    query_times = []
    
    # Store results for each metric
    # TODO: Update to `ir_metric_results`....
    metric_results = {metric.__name__: [] for metric in metrics}
    
    for i, (result, ground_truth) in enumerate(tqdm(zip(retrieved_ids, ground_truths))):
        # Skip if there was an error
        # TODO: Pretty sure this isn't setup
        if "error" in result:
            print(f"\n\033[91mSkipping analysis for query {i} due to error: {result['error']}\033[0m")
            for metric in metrics:
                metric_results[metric.__name__].append(0.0)
            query_times.append(result["time_taken"])
            continue
        
        for metric in metrics:
            if metric.__name__ == "calculate_recall":
                # Traditional recall - just use target IDs
                score = metric(
                    ground_truth["dataset_ids"],
                    retrieved_ids
                )
            elif metric.__name__ in ["calculate_coverage", "calculate_alpha_ndcg"]:
                # Ensure nuggets have IDs
                if ground_truth.get("nugget_data"):
                    for idx, nugget in enumerate(ground_truth["nugget_data"]):
                        if 'id' not in nugget:
                            nugget['id'] = f"nugget_{idx}"
                
                if metric.__name__ == "calculate_coverage":
                    score = metric(retrieved_ids, ground_truth["nugget_data"], k=1000)
                else:  # calculate_alpha_ndcg
                    score = metric(retrieved_ids, ground_truth["nugget_data"], alpha=0.5, k=10)
            
            metric_results[metric.__name__].append(score)
        
        # Store other metrics
        query_times.append(result["time_taken"])
        
        # Print rolling update every 10 queries
        if (i + 1) % 10 == 0:
            print(f"\n\033[93m--- Analysis Progress ({i + 1}/{len(retrieved_ids)}) ---\033[0m")
            for metric_name, scores in metric_results.items():
                if scores:
                    # Clean up metric name for display
                    display_name = metric_name.replace("calculate_", "").replace("_", " ").title()
                    print(f"Current average {display_name}: {np.mean(scores):.2f}")
            
            print(f"Current average query time: {np.mean(query_times):.2f} seconds")
    
    # Build results dictionary
    results_dict = {
        "avg_query_time": np.mean(query_times) if query_times else 0,
        "query_times": query_times,
    }
    
    # Add metric-specific results
    for metric_name, scores in metric_results.items():
        clean_name = metric_name.replace("calculate_", "")
        results_dict[f"avg_{clean_name}"] = np.mean(scores) if scores else 0
        results_dict[f"{clean_name}_scores"] = scores
    
    # Print summary
    print("\n\033[92m===== Benchmark Results =====\033[0m")
    print(f"Dataset: {dataset_name}")
    print(f"Number of queries: {len(retrieved_ids)}")
    
    # Print metric results
    for metric_name, scores in metric_results.items():
        if scores:
            display_name = metric_name.replace("calculate_", "").replace("_", " ").title()
            print(f"Average {display_name}: {np.mean(scores):.2f}")
    
    print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")
    
    return results_dict