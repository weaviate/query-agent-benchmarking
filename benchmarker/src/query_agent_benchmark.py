import time
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

async def analyze_results(
    weaviate_client: Any,
    dataset_name: str,
    results: list,
    ground_truths: list[dict],
    judge_inferences: int = 3,
):
    if dataset_name == "enron":
        collection = weaviate_client.collections.get("EnronEmails")
    elif dataset_name == "wixqa":
        collection = weaviate_client.collections.get("WixKB")
    
    lm_judge_average_scores = []
    lm_judge_score_ranges = []
    lm_judge_score_variances = []
    recall_scores = []
    query_times = []
    
    for i, (result, ground_truth) in enumerate(tqdm(zip(results, ground_truths), desc="Analyzing results", total=len(results))):
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
                print(f"Latest LM Judge score: N/A (empty answer)")
                print(f"Latest LM Judge score range: N/A (empty answer)")
                print(f"Latest LM Judge score variance: N/A (empty answer)")
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


def query_agent_benchmark_metrics_to_markdown(metrics: dict, dataset_name: str = None, experiment_name: str = None, output_path: str = None):
    """
    Formats the metrics returned by analyze_results function as markdown and saves to disk.
    
    Args:
        metrics: Dictionary containing benchmark metrics
        dataset_name: Optional name of the dataset used
        experiment_name: Optional name of the experiment
        output_path: Path to save the markdown file (defaults to results/metrics_{timestamp}.md)
    """
    import os
    from datetime import datetime
    
    # Generate markdown content
    markdown = []
    
    # Add header with experiment info
    if experiment_name:
        markdown.append(f"# {experiment_name} Benchmark Results")
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
        exp_name = experiment_name.replace(" ", "_").lower() if experiment_name else "query_agent"
        output_path = os.path.join(results_dir, f"metrics_{exp_name}_{file_timestamp}.md")
    
    # Write markdown to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(markdown))
    
    print(f"Metrics saved as markdown to {output_path}")
    return output_path