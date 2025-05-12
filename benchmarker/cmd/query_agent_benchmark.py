import time
from typing import Any
from benchmarker.cmd.metrics.lm_as_judge_agent import lm_as_judge_agent, LMJudgeAgentDeps
from benchmarker.cmd.metrics.ir_metrics import calculate_recall_at_k
from benchmarker.cmd.utils import qa_source_parser

def run_queries(
    queries: list[str],
    query_agent: callable,
    test_samples: int
):
    results = []
    start = time.time()
    for query in queries[test_samples]:
        response = query_agent.run(query["question"])
        results.append({
            "query": query,
            "query_id": query["dataset_id"],
            "answer": response.final_answer,
            "sources": response.sources,
            "misc_response": response,
        })
    print(f"Experiment completed {len(queries)} in {time.time() - start} seconds.")
    return results

def analyze_results(
    weaviate_client: Any,
    dataset_name: str,
    results: list,
    ground_truths: str,
):
    if dataset_name == "enron":
            collection = weaviate_client.collections.get("EnronEmails")
    
    for result, ground_truth in zip(results, ground_truths):
        # calculate lm_as_judge score
        deps=LMJudgeAgentDeps(
            question=ground_truth["question"],
            system_response=results["answer"]
        )
        lm_as_judge_agent.run(
            deps=deps,
            model="openai:gpt-4.1"
        )
        # calculate recall
        source_objects = qa_source_parser(
             result["sources"],
             collection
        )
        calculate_recall_at_k(
            source_objects,
            ground_truth["dataset_id"]
        )