import time
import asyncio
from typing import Any
from tqdm import tqdm
import numpy as np
from benchmarker.src.metrics.lm_as_judge_agent import lm_as_judge_agent, LMJudgeAgentDeps
from benchmarker.src.metrics.ir_metrics import calculate_recall
from benchmarker.src.utils import qa_source_parser

def run_queries(
    queries: list[dict],
    query_agent: Any,
    num_samples: int
):
    """Synchronous version of run_queries"""
    results = []
    start = time.time()
    for i, query in enumerate(tqdm(queries[:num_samples], desc="Running queries")):
        query_start_time = time.time()
        response = query_agent.run(query["question"])
        query_time_taken = time.time() - query_start_time

        results.append({
            "query": query,
            "query_id": query["dataset_ids"],
            "answer": response.final_answer,
            "sources": response.sources,
            "num_searches": len(response.searches),
            "num_aggregations": len(response.aggregations),
            "misc_response": response,
            "time_taken": query_time_taken
        })
        
        # Print rolling update every 10 queries
        if (i + 1) % 10 == 0:
            print(f"\n\033[93m--- Progress Update ({i + 1}/{num_samples}) ---\033[0m")
            print(f"Latest query: {query['question']}")
            print(f"Latest response: {response.final_answer[:200]}...")
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
    
    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_query(query, index, retry_count=0, max_retries=3):
        async with semaphore:
            query_start_time = time.time()
            try:
                # Add a small delay between requests to avoid overwhelming the API
                if retry_count > 0:
                    delay = min(2 ** retry_count, 10)  # Exponential backoff, max 10 seconds
                    print(f"\nRetrying query {index} (attempt {retry_count + 1}) after {delay}s delay...")
                    await asyncio.sleep(delay)
                elif index > 0:
                    # Small delay between requests to be respectful to the API
                    await asyncio.sleep(0.1)
                
                response = await query_agent.run_async(query["question"])
                query_time_taken = time.time() - query_start_time
                
                return {
                    "query": query,
                    "query_id": query["dataset_ids"],
                    "answer": response.final_answer,
                    "sources": response.sources,
                    "num_searches": len(response.searches),
                    "num_aggregations": len(response.aggregations),
                    "misc_response": response,
                    "time_taken": query_time_taken,
                    "index": index  # Keep track of original order
                }
            except Exception as e:
                error_msg = str(e)
                query_time_taken = time.time() - query_start_time
                
                # Check if it's a connection error that might benefit from retry
                if "connection" in error_msg.lower() and retry_count < max_retries:
                    print(f"\nConnection error for query {index}, retrying: {error_msg}")
                    return await process_query(query, index, retry_count + 1, max_retries)
                
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
    
    # Process queries in batches
    queries_to_process = queries[:num_samples]
    total_batches = (len(queries_to_process) + batch_size - 1) // batch_size
    
    print(f"\033[94mProcessing {len(queries_to_process)} queries in {total_batches} batches "
          f"(batch_size={batch_size}, max_concurrent={max_concurrent})\033[0m")
    
    for batch_idx in range(0, len(queries_to_process), batch_size):
        batch = queries_to_process[batch_idx:batch_idx + batch_size]
        batch_start = time.time()
        
        print(f"\nStarting batch {batch_idx // batch_size + 1}/{total_batches}")
        
        # Create tasks for this batch
        tasks = [
            process_query(query, batch_idx + i) 
            for i, query in enumerate(batch)
        ]
        
        # Process batch concurrently
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        batch_time = time.time() - batch_start
        completed = min(batch_idx + batch_size, len(queries_to_process))
        
        # Count successes and errors
        batch_successes = sum(1 for r in batch_results if "error" not in r)
        batch_errors = len(batch_results) - batch_successes
        
        print(f"\n\033[93m--- Batch {batch_idx // batch_size + 1} Complete ({completed}/{len(queries_to_process)}) ---\033[0m")
        print(f"Successes: {batch_successes}, Errors: {batch_errors}")
        print(f"Batch completed in {batch_time:.2f} seconds")
        
        # Print sample from successful queries
        successful_results = [r for r in batch_results if "error" not in r]
        if successful_results:
            sample = successful_results[0]
            print(f"Sample query: {sample['query']['question'][:100]}...")
            print(f"Sample response: {sample['answer'][:200]}...")
        
        # Add a small delay between batches
        if batch_idx + batch_size < len(queries_to_process):
            await asyncio.sleep(1)
    
    # Sort results by original index to maintain order
    results.sort(key=lambda x: x["index"])
    
    # Remove index field as it's no longer needed
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

async def analyze_results(
    weaviate_client: Any,
    dataset_name: str,
    results: list,
    ground_truths: list[dict],
    judge_inferences: int = 3,
):
    """Analyze results (remains mostly the same, already async)"""
    if dataset_name == "enron":
        collection = weaviate_client.collections.get("EnronEmails")
    elif dataset_name == "wixqa":
        collection = weaviate_client.collections.get("WixKB")
    elif dataset_name == "freshstack-langchain":
        collection = weaviate_client.collections.get("FreshStackLangChain")
    else:
        raise Exception("Enter a valid dataset_name!")
    
    lm_judge_average_scores = []
    lm_judge_score_ranges = []
    lm_judge_score_variances = []
    recall_scores = []
    query_times = []
    
    for i, (result, ground_truth) in enumerate(tqdm(zip(results, ground_truths), desc="Analyzing results", total=len(results))):
        # Skip if there was an error
        if "error" in result:
            print(f"\n\033[91mSkipping analysis for query {i} due to error: {result['error']}\033[0m")
            lm_judge_average_scores.append(0.0)
            lm_judge_score_ranges.append(0.0)
            lm_judge_score_variances.append(0.0)
            recall_scores.append(0.0)
            query_times.append(result["time_taken"])
            continue
            
        # calculate lm_as_judge score
        if result["answer"] == "":
            lm_judge_average_scores.append(1.0)
            lm_judge_score_ranges.append(0.0)
            lm_judge_score_variances.append(0.0)
        else:
            deps = LMJudgeAgentDeps(
                question=ground_truth["question"],
                system_response=result["answer"]
            )

            # Run judge multiple times and collect scores
            current_scores = []
            for _ in range(judge_inferences):
                judge_response = await lm_as_judge_agent.run(
                    deps=deps,
                    model="openai:gpt-4.1"
                )
                current_scores.append(judge_response.data.rating)
            
            # Calculate statistics
            avg_score = sum(current_scores) / len(current_scores)
            score_range = max(current_scores) - min(current_scores)
            score_variance = np.var(current_scores)
            
            # Store the statistics
            lm_judge_average_scores.append(avg_score)
            lm_judge_score_ranges.append(score_range)
            lm_judge_score_variances.append(score_variance)

        # calculate recall
        sources_counter = len(result["sources"])
        print(f"Starting with {sources_counter} sources!")
        source_objects = qa_source_parser(
            result["sources"],
            collection
        )
        print(f"Found {len(source_objects)} to compute recall against!")
        
        if dataset_name == "freshstack-langchain":
            recall = calculate_recall(
                ground_truth["dataset_ids"],
                source_objects,
                nugget_data=ground_truth["nugget_data"]
            )
        else:
            recall = calculate_recall(
                ground_truth["dataset_ids"],
                source_objects
            )
        recall_scores.append(recall)
        
        # Store query time
        query_times.append(result["time_taken"])
        
        # Print rolling update every 10 queries
        if (i + 1) % 10 == 0:
            current_avg_recall = np.mean(recall_scores)
            current_avg_lm_judge = np.mean(lm_judge_average_scores)
            current_avg_range = np.mean(lm_judge_score_ranges)
            current_avg_variance = np.mean(lm_judge_score_variances)
            current_avg_time = np.mean(query_times)
            print(f"\n\033[93m--- Analysis Progress ({i + 1}/{len(results)}) ---\033[0m")
            print(f"Current average recall: {current_avg_recall:.2f}")
            print(f"Current average LM Judge score: {current_avg_lm_judge:.2f}")
            print(f"Current average LM Judge score range: {current_avg_range:.2f}")
            print(f"Current average LM Judge score variance: {current_avg_variance:.4f}")
            print(f"Current average query time: {current_avg_time:.2f} seconds")
            print(f"Latest recall: {recall:.2f}")
            if result["answer"] == "":
                print("Latest LM Judge score: N/A (empty answer)")
                print("Latest LM Judge score range: N/A (empty answer)")
                print("Latest LM Judge score variance: N/A (empty answer)")
            else:
                print(f"Latest LM Judge score: {avg_score:.2f}")
                print(f"Latest LM Judge score range: {score_range:.2f}")
                print(f"Latest LM Judge score variance: {score_variance:.4f}")
            print(f"Latest query time: {result['time_taken']:.2f} seconds")
    
    # Calculate aggregate metrics
    avg_lm_judge_score = sum(lm_judge_average_scores) / len(lm_judge_average_scores) if lm_judge_average_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_lm_judge_range = sum(lm_judge_score_ranges) / len(lm_judge_score_ranges) if lm_judge_score_ranges else 0
    avg_lm_judge_variance = sum(lm_judge_score_variances) / len(lm_judge_score_variances) if lm_judge_score_variances else 0
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    
    # Print summary
    print("\033[92m\n===== Benchmark Results =====\033[0m")
    print(f"Dataset: {dataset_name}")
    print(f"Number of queries: {len(results)}")
    print(f"Average LM Judge Score: {avg_lm_judge_score:.2f}")
    print(f"Average LM Judge Score Range: {avg_lm_judge_range:.2f}")
    print(f"Average LM Judge Score Variance: {avg_lm_judge_variance:.4f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average Query Time: {avg_query_time:.2f} seconds")
    
    return {
        "avg_lm_judge_score": avg_lm_judge_score,
        "avg_lm_judge_range": avg_lm_judge_range,
        "avg_lm_judge_variance": avg_lm_judge_variance,
        "avg_recall": avg_recall,
        "avg_query_time": avg_query_time,
        "lm_judge_average_scores": lm_judge_average_scores,
        "lm_judge_score_ranges": lm_judge_score_ranges,
        "lm_judge_score_variances": lm_judge_score_variances,
        "recall_scores": recall_scores,
        "query_times": query_times
    }


def pretty_print_query_agent_benchmark_metrics(metrics: dict, dataset_name: str = None, experiment_name: str = None):
    """
    Pretty prints the metrics returned by analyze_results function.
    
    Args:
        metrics: Dictionary containing benchmark metrics
        dataset_name: Optional name of the dataset used
        experiment_name: Optional name of the experiment
    """
    # Print header with experiment info if provided
    print("\n" + "="*60)
    if experiment_name:
        print(f"\033[1m\033[94m{experiment_name} Benchmark Results\033[0m")
    else:
        print("\033[1m\033[94mQuery Agent Benchmark Results\033[0m")
    print("="*60)
    
    # Print dataset info if provided
    if dataset_name:
        print(f"\033[1mDataset:\033[0m {dataset_name}")
    
    # Print aggregate metrics with formatting
    print("\n\033[1m\033[92mAggregate Metrics:\033[0m")
    print(f"  • Average LM Judge Score:       \033[1m{metrics['avg_lm_judge_score']:.2f}\033[0m")
    print(f"  • Average Recall:               \033[1m{metrics['avg_recall']:.2f}\033[0m")
    print(f"  • Average LM Judge Score Range: \033[1m{metrics['avg_lm_judge_range']:.2f}\033[0m")
    print(f"  • Average LM Judge Variance:    \033[1m{metrics['avg_lm_judge_variance']:.4f}\033[0m")
    print(f"  • Average Query Time:           \033[1m{metrics['avg_query_time']:.2f}\033[0m seconds")
    
    # Print distribution statistics if available
    if 'lm_judge_average_scores' in metrics and metrics['lm_judge_average_scores']:
        print("\n\033[1m\033[92mDistribution Statistics:\033[0m")
        
        # LM Judge scores
        lm_scores = np.array(metrics['lm_judge_average_scores'])
        print("  • LM Judge Scores:")
        print(f"    - Min: {np.min(lm_scores):.2f}")
        print(f"    - Max: {np.max(lm_scores):.2f}")
        print(f"    - Median: {np.median(lm_scores):.2f}")
        print(f"    - Std Dev: {np.std(lm_scores):.2f}")
        
        # Recall scores
        recall_scores = np.array(metrics['recall_scores'])
        print("  • Recall Scores:")
        print(f"    - Min: {np.min(recall_scores):.2f}")
        print(f"    - Max: {np.max(recall_scores):.2f}")
        print(f"    - Median: {np.median(recall_scores):.2f}")
        print(f"    - Std Dev: {np.std(recall_scores):.2f}")
        
        # Query times
        if 'query_times' in metrics and metrics['query_times']:
            query_times = np.array(metrics['query_times'])
            print("  • Query Times (seconds):")
            print(f"    - Min: {np.min(query_times):.2f}")
            print(f"    - Max: {np.max(query_times):.2f}")
            print(f"    - Median: {np.median(query_times):.2f}")
            print(f"    - Std Dev: {np.std(query_times):.2f}")
    
    print("\n" + "="*60)


def query_agent_benchmark_metrics_to_markdown(metrics: dict, dataset_name: str = None, agent_name: str = None, output_path: str = None):
    """
    Formats the metrics returned by analyze_results function as markdown and saves to disk.
    
    Args:
        metrics: Dictionary containing benchmark metrics
        dataset_name: Optional name of the dataset used
        agent_name: Optional name of the agent
        output_path: Path to save the markdown file (defaults to results/metrics_{timestamp}.md)
    """
    import os
    from datetime import datetime
    
    # Generate markdown content
    markdown = []
    
    # Add header with agent info
    if agent_name:
        markdown.append(f"# {agent_name} Benchmark Results")
    else:
        markdown.append("# Query Agent Benchmark Results")
    
    # Add dataset info if provided
    if dataset_name:
        markdown.append(f"\n**Dataset:** {dataset_name}")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown.append(f"\n**Generated:** {timestamp}")
    
    # Add aggregate metrics
    markdown.append("\n## Aggregate Metrics")
    markdown.append("| Metric | Value |")
    markdown.append("| ------ | ----- |")
    markdown.append(f"| Average LM Judge Score | **{metrics['avg_lm_judge_score']:.2f}** |")
    markdown.append(f"| Average Recall | **{metrics['avg_recall']:.2f}** |")
    markdown.append(f"| Average LM Judge Score Range | **{metrics['avg_lm_judge_range']:.2f}** |")
    markdown.append(f"| Average LM Judge Variance | **{metrics['avg_lm_judge_variance']:.4f}** |")
    markdown.append(f"| Average Query Time | **{metrics['avg_query_time']:.2f}** seconds |")
    
    # Add distribution statistics if available
    if 'lm_judge_average_scores' in metrics and metrics['lm_judge_average_scores']:
        markdown.append("\n## Distribution Statistics")
        
        # LM Judge scores
        lm_scores = np.array(metrics['lm_judge_average_scores'])
        markdown.append("\n### LM Judge Scores")
        markdown.append("| Statistic | Value |")
        markdown.append("| --------- | ----- |")
        markdown.append(f"| Min | {np.min(lm_scores):.2f} |")
        markdown.append(f"| Max | {np.max(lm_scores):.2f} |")
        markdown.append(f"| Median | {np.median(lm_scores):.2f} |")
        markdown.append(f"| Std Dev | {np.std(lm_scores):.2f} |")
        
        # Recall scores
        recall_scores = np.array(metrics['recall_scores'])
        markdown.append("\n### Recall Scores")
        markdown.append("| Statistic | Value |")
        markdown.append("| --------- | ----- |")
        markdown.append(f"| Min | {np.min(recall_scores):.2f} |")
        markdown.append(f"| Max | {np.max(recall_scores):.2f} |")
        markdown.append(f"| Median | {np.median(recall_scores):.2f} |")
        markdown.append(f"| Std Dev | {np.std(recall_scores):.2f} |")
        
        # Query times
        if 'query_times' in metrics and metrics['query_times']:
            query_times = np.array(metrics['query_times'])
            markdown.append("\n### Query Times (seconds)")
            markdown.append("| Statistic | Value |")
            markdown.append("| --------- | ----- |")
            markdown.append(f"| Min | {np.min(query_times):.2f} |")
            markdown.append(f"| Max | {np.max(query_times):.2f} |")
            markdown.append(f"| Median | {np.median(query_times):.2f} |")
            markdown.append(f"| Std Dev | {np.std(query_times):.2f} |")
    
    # Determine output path
    if output_path is None:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename with timestamp
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name_clean = agent_name.replace(" ", "_").lower() if agent_name else "query_agent"
        output_path = os.path.join(results_dir, f"metrics_{agent_name_clean}_{file_timestamp}.md")
    
    # Write markdown to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(markdown))
    
    print(f"Metrics saved as markdown to {output_path}")
    return output_path