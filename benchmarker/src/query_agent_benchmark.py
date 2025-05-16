import time
from typing import Any
from tqdm import tqdm
import numpy as np
from benchmarker.cmd.metrics.lm_as_judge_agent import lm_as_judge_agent, LMJudgeAgentDeps
from benchmarker.cmd.metrics.ir_metrics import calculate_recall
from benchmarker.cmd.utils import qa_source_parser

def run_queries(
    queries: list[dict],
    query_agent,
    test_samples: int
):
    results = []
    start = time.time()
    for i, query in enumerate(tqdm(queries[:test_samples], desc="Running queries")):
        response = query_agent.run(query["question"])
        results.append({
            "query": query,
            "query_id": query["dataset_ids"],
            "answer": response.final_answer,
            "sources": response.sources,
            "misc_response": response,
        })
        
        # Print rolling update every 10 queries
        if (i + 1) % 10 == 0:
            print(f"\n\033[93m--- Progress Update ({i + 1}/{test_samples}) ---\033[0m")
            print(f"Latest query: {query['question']}")
            print(f"Latest response: {response.final_answer[:200]}...")
            
    print(f"Experiment completed {len(results)} queries in {time.time() - start:.2f} seconds.")
    return results

async def analyze_results(
    weaviate_client: Any,
    dataset_name: str,
    results: list,
    ground_truths: list[dict],
):
    if dataset_name == "enron":
        collection = weaviate_client.collections.get("EnronEmails")
    elif dataset_name == "wixqa":
        collection = weaviate_client.collections.get("WixKB")
    
    lm_judge_scores = []
    recall_scores = []
    
    for i, (result, ground_truth) in enumerate(tqdm(zip(results, ground_truths), desc="Analyzing results", total=len(results))):
        # calculate lm_as_judge score
        deps = LMJudgeAgentDeps(
            question=ground_truth["question"],
            system_response=result["answer"]
        )
        judge_response = await lm_as_judge_agent.run(
            deps=deps,
            model="openai:gpt-4.1"
        )
        lm_judge_scores.append(judge_response.data.rating)
        
        # calculate recall
        source_objects = qa_source_parser(
             result["sources"],
             collection
        )
        recall = calculate_recall(
            ground_truth["dataset_ids"],
            source_objects
        )
        recall_scores.append(recall)
        
        # Print rolling update every 10 queries
        if (i + 1) % 10 == 0:
            current_avg_recall = np.mean(recall_scores)
            current_avg_lm_judge = np.mean(lm_judge_scores)
            print(f"\n\033[93m--- Analysis Progress ({i + 1}/{len(results)}) ---\033[0m")
            print(f"Current average recall: {current_avg_recall:.2f}")
            print(f"Current average LM Judge score: {current_avg_lm_judge:.2f}")
            print(f"Latest recall: {recall:.2f}")
            print(f"Latest LM Judge score: {judge_response.data.rating:.2f}")
    
    # Calculate aggregate metrics
    avg_lm_judge_score = sum(lm_judge_scores) / len(lm_judge_scores) if lm_judge_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    
    # Print summary
    print("\033[92m\n===== Benchmark Results =====\033[0m")
    print(f"Dataset: {dataset_name}")
    print(f"Number of queries: {len(results)}")
    print(f"Average LM Judge Score: {avg_lm_judge_score:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    
    return {
        "results": results,
        "metrics": {
            "avg_lm_judge_score": avg_lm_judge_score,
            "avg_recall": avg_recall,
            "lm_judge_scores": lm_judge_scores,
            "recall_scores": recall_scores
        }
    }