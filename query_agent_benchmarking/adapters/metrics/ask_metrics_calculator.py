"""AskMetricsCalculator adapter implementations.

Provides calculators for different ask evaluation strategies:
- LMJudgeAskCalculator: Uses LLM-as-judge (DSPy) for semantic alignment
- ExactMatchAskCalculator: Uses exact string matching
- OfficeQAAskCalculator: Uses fuzzy numerical matching
"""

import numpy as np
from tqdm import tqdm

from query_agent_benchmarking.domain.models import AskResult
from query_agent_benchmarking.metrics.lmjudge_alignment import LMJudge
from query_agent_benchmarking.metrics.exact_match import calculate_exact_match
from query_agent_benchmarking.metrics.officeqa_metric import (
    score_answer as officeqa_score_answer,
    extract_final_answer,
)


class LMJudgeAskCalculator:
    """Ask metrics calculator using LLM-as-judge for semantic alignment."""

    def __init__(self, model: str = "openai/gpt-4.1", ensemble_k: int = 3):
        self.model = model
        self.ensemble_k = ensemble_k
        self.judge = LMJudge(model=model, ensemble_k=ensemble_k)

    def compute(self, results: list[AskResult]) -> dict:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with LLM judge...\033[0m")
        print(f"Judge model: {self.model}, Ensemble K: {self.ensemble_k}")

        alignment_scores = []
        query_times = []
        misaligned_indices = []
        total_input_tokens = 0
        total_output_tokens = 0

        for i, result in enumerate(tqdm(results, desc="Running LLM judge")):
            if result.system_answer.startswith("[ERROR]"):
                print(f"\n\033[91mSkipping evaluation for query {i} due to error.\033[0m")
                continue

            judge_result = self.judge.evaluate_with_details(
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

            if (i + 1) % 5 == 0:
                current_avg = np.mean(alignment_scores) if alignment_scores else 0
                print(f"\n\033[93m--- Evaluation Progress ({i + 1}/{len(results)}) ---\033[0m")
                print(f"Running alignment score: {current_avg:.2%}")
                print(f"Running avg query time: {np.mean(query_times):.2f}s")

        results_dict = {
            "avg_query_time": np.mean(query_times) if query_times else 0,
            "query_times": query_times,
            "misaligned_indices": misaligned_indices,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "metric": "llm_judge",
            "avg_alignment_score": np.mean(alignment_scores) if alignment_scores else 0,
            "alignment_score_scores": alignment_scores,
            "judge_model": self.model,
            "ensemble_k": self.ensemble_k,
        }

        self._print_summary(alignment_scores, results_dict, misaligned_indices,
                            total_input_tokens, total_output_tokens)
        return results_dict

    @staticmethod
    def _print_summary(alignment_scores, results_dict, misaligned_indices,
                       total_input_tokens, total_output_tokens):
        print("\n\033[92m===== Ask Benchmark Results =====\033[0m")
        print(f"Number of queries evaluated: {len(alignment_scores)}")
        if alignment_scores:
            print(f"Average Alignment Score: {np.mean(alignment_scores):.2%}")
        print(f"Misaligned queries: {len(misaligned_indices)}")
        print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")
        print(f"Total Judge Tokens: {total_input_tokens + total_output_tokens}")


class ExactMatchAskCalculator:
    """Ask metrics calculator using exact string matching."""

    def compute(self, results: list[AskResult]) -> dict:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with exact match...\033[0m")

        alignment_scores = []
        query_times = []
        misaligned_indices = []

        for i, result in enumerate(tqdm(results, desc="Running exact match")):
            if result.system_answer.startswith("[ERROR]"):
                continue

            aligned = calculate_exact_match(
                system_answer=result.system_answer,
                ground_truth_answer=result.query.ground_truth_answer,
            )
            alignment_scores.append(1 if aligned else 0)
            query_times.append(result.time_taken)
            if not aligned:
                misaligned_indices.append(i)

        results_dict = {
            "avg_query_time": np.mean(query_times) if query_times else 0,
            "query_times": query_times,
            "misaligned_indices": misaligned_indices,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "metric": "exact_match",
            "avg_exact_match_accuracy": np.mean(alignment_scores) if alignment_scores else 0,
            "exact_match_accuracy_scores": alignment_scores,
        }

        print("\n\033[92m===== Ask Benchmark Results =====\033[0m")
        print(f"Number of queries evaluated: {len(alignment_scores)}")
        if alignment_scores:
            print(f"Average Exact Match Accuracy: {np.mean(alignment_scores):.2%}")
        print(f"Misaligned queries: {len(misaligned_indices)}")
        print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")

        return results_dict


class OfficeQAAskCalculator:
    """Ask metrics calculator using OfficeQA fuzzy numerical matching."""

    def __init__(self, tolerance: float = 0.00):
        self.tolerance = tolerance

    def compute(self, results: list[AskResult]) -> dict:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with OfficeQA fuzzy match "
              f"(tolerance={self.tolerance})...\033[0m")

        alignment_scores = []
        query_times = []
        misaligned_indices = []

        for i, result in enumerate(tqdm(results, desc="Running OfficeQA fuzzy match")):
            if result.system_answer.startswith("[ERROR]"):
                continue

            predicted = extract_final_answer(result.system_answer)
            score = officeqa_score_answer(
                ground_truth=result.query.ground_truth_answer,
                predicted=predicted,
                tolerance=self.tolerance,
            )
            aligned = score == 1.0
            alignment_scores.append(1 if aligned else 0)
            query_times.append(result.time_taken)
            if not aligned:
                misaligned_indices.append(i)

        results_dict = {
            "avg_query_time": np.mean(query_times) if query_times else 0,
            "query_times": query_times,
            "misaligned_indices": misaligned_indices,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "metric": "officeqa_fuzzy_match",
            "avg_officeqa_accuracy": np.mean(alignment_scores) if alignment_scores else 0,
            "officeqa_accuracy_scores": alignment_scores,
        }

        print("\n\033[92m===== Ask Benchmark Results =====\033[0m")
        print(f"Number of queries evaluated: {len(alignment_scores)}")
        if alignment_scores:
            print(f"Average OfficeQA Accuracy: {np.mean(alignment_scores):.2%}")
        print(f"Misaligned queries: {len(misaligned_indices)}")
        print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")

        return results_dict
