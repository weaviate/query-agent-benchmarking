"""AskMetricsCalculator adapter implementations.

Provides calculators for different ask evaluation strategies:
- LMJudgeAskCalculator: Uses LLM-as-judge (DSPy) for semantic alignment
- ExactMatchAskCalculator: Uses exact string matching
- OfficeQAAskCalculator: Uses fuzzy numerical matching
- LongMemEvalAskCalculator: Uses LongMemEval's type-specific LLM judge
- QueryExactMatchAskCalculator: Exact match of the database queries the agent issued (QueryAgentGym)
"""

from typing import Optional

import numpy as np
from tqdm import tqdm

from query_agent_benchmarking.internal.core.domain.models import AskResult
from query_agent_benchmarking.internal.adapters.metrics.lmjudge_alignment import LMJudge
from query_agent_benchmarking.internal.adapters.metrics.longmemeval_judge import LongMemEvalJudge
from query_agent_benchmarking.internal.adapters.metrics.exact_match import calculate_exact_match
from query_agent_benchmarking.internal.adapters.metrics.query_exact_match import (
    canonicalize_ground_truth,
    canonicalize_predicted,
    render_unit,
    score_query_units,
)
from query_agent_benchmarking.internal.adapters.metrics.officeqa_metric import (
    score_answer as officeqa_score_answer,
    extract_final_answer,
)


class LMJudgeAskCalculator:
    """Ask metrics calculator using LLM-as-judge for semantic alignment."""

    def __init__(self, model: str = "openai/gpt-4.1", ensemble_k: int = 3, use_reasoning: bool = False):
        self.model = model
        self.ensemble_k = ensemble_k
        self.use_reasoning = use_reasoning
        self.judge = LMJudge(model=model, ensemble_k=ensemble_k)

    def compute(self, results: list[AskResult]) -> dict:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with LLM judge...\033[0m")
        print(f"Judge model: {self.model}, Ensemble K: {self.ensemble_k}")

        alignment_scores = []
        judge_reasonings = []
        query_times = []
        misaligned_indices = []
        total_input_tokens = 0
        total_output_tokens = 0

        for i, result in enumerate(tqdm(results, desc="Running LLM judge")):
            if result.system_answer.startswith("[ERROR]"):
                print(f"\n\033[91mSkipping evaluation for query {i} due to error.\033[0m")
                alignment_scores.append(None)
                if self.use_reasoning:
                    judge_reasonings.append(None)
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

            if self.use_reasoning:
                judge_reasonings.append(judge_result.get("reasoning"))

            if not aligned:
                misaligned_indices.append(i)

            if (i + 1) % 5 == 0:
                current_avg = np.mean(alignment_scores) if alignment_scores else 0
                print(f"\n\033[93m--- Evaluation Progress ({i + 1}/{len(results)}) ---\033[0m")
                print(f"Running alignment score: {current_avg:.2%}")
                print(f"Running avg query time: {np.mean(query_times):.2f}s")

        valid_scores = [s for s in alignment_scores if s is not None]
        results_dict = {
            "avg_query_time": np.mean(query_times) if query_times else 0,
            "query_times": query_times,
            "misaligned_indices": misaligned_indices,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "metric": "llm_judge",
            "avg_alignment_score": float(np.mean(valid_scores)) if valid_scores else 0,
            "alignment_score_scores": alignment_scores,
            "judge_model": self.model,
            "ensemble_k": self.ensemble_k,
        }

        if self.use_reasoning:
            results_dict["judge_reasonings"] = judge_reasonings

        self._print_summary(valid_scores, results_dict, misaligned_indices,
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
                alignment_scores.append(None)
                continue

            aligned = calculate_exact_match(
                system_answer=result.system_answer,
                ground_truth_answer=result.query.ground_truth_answer,
            )
            alignment_scores.append(1 if aligned else 0)
            query_times.append(result.time_taken)
            if not aligned:
                misaligned_indices.append(i)

        valid_scores = [s for s in alignment_scores if s is not None]
        results_dict = {
            "avg_query_time": np.mean(query_times) if query_times else 0,
            "query_times": query_times,
            "misaligned_indices": misaligned_indices,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "metric": "exact_match",
            "avg_exact_match_accuracy": float(np.mean(valid_scores)) if valid_scores else 0,
            "exact_match_accuracy_scores": alignment_scores,
        }

        print("\n\033[92m===== Ask Benchmark Results =====\033[0m")
        print(f"Number of queries evaluated: {len(valid_scores)}")
        if valid_scores:
            print(f"Average Exact Match Accuracy: {np.mean(valid_scores):.2%}")
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
                alignment_scores.append(None)
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

        valid_scores = [s for s in alignment_scores if s is not None]
        results_dict = {
            "avg_query_time": np.mean(query_times) if query_times else 0,
            "query_times": query_times,
            "misaligned_indices": misaligned_indices,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "metric": "officeqa_fuzzy_match",
            "avg_officeqa_accuracy": float(np.mean(valid_scores)) if valid_scores else 0,
            "officeqa_accuracy_scores": alignment_scores,
        }

        print("\n\033[92m===== Ask Benchmark Results =====\033[0m")
        print(f"Number of queries evaluated: {len(valid_scores)}")
        if valid_scores:
            print(f"Average OfficeQA Accuracy: {np.mean(valid_scores):.2%}")
        print(f"Misaligned queries: {len(misaligned_indices)}")
        print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")

        return results_dict


class LongMemEvalAskCalculator:
    """Ask metrics calculator using LongMemEval's type-specific LLM judge.

    Uses separate GPT-4o prompts per question type (temporal-reasoning,
    knowledge-update, single-session-preference, etc.) following the
    evaluation protocol from the LongMemEval paper (Wu et al., ICLR 2025).
    """

    def __init__(self, model: str = "openai/gpt-5.4", api_key: str | None = None):
        self.model = model
        self.judge = LongMemEvalJudge(model=model, api_key=api_key)

    def compute(self, results: list[AskResult]) -> dict:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with LongMemEval judge...\033[0m")
        print(f"Judge model: {self.model}")

        alignment_scores = []
        query_times = []
        misaligned_indices = []
        total_input_tokens = 0
        total_output_tokens = 0
        type_scores: dict[str, list[int]] = {}

        for i, result in enumerate(tqdm(results, desc="Running LongMemEval judge")):
            if result.system_answer.startswith("[ERROR]"):
                print(f"\n\033[91mSkipping evaluation for query {i} due to error.\033[0m")
                alignment_scores.append(None)
                continue

            qtype = result.query.question_type or "multi-session"
            judge_result = self.judge.evaluate_with_details(
                question=result.query.question,
                system_answer=result.system_answer,
                correct_answer=result.query.ground_truth_answer,
                question_type=qtype,
            )

            aligned = judge_result["aligned"]
            score = 1 if aligned else 0
            alignment_scores.append(score)
            query_times.append(result.time_taken)
            total_input_tokens += judge_result.get("input_tokens", 0)
            total_output_tokens += judge_result.get("output_tokens", 0)

            type_scores.setdefault(qtype, []).append(score)

            if not aligned:
                misaligned_indices.append(i)

            if (i + 1) % 5 == 0:
                valid_so_far = [s for s in alignment_scores if s is not None]
                current_avg = np.mean(valid_so_far) if valid_so_far else 0
                print(f"\n\033[93m--- Evaluation Progress ({i + 1}/{len(results)}) ---\033[0m")
                print(f"Running accuracy: {current_avg:.2%}")
                print(f"Running avg query time: {np.mean(query_times):.2f}s")

        # Per-type accuracy breakdown
        type_accuracy = {
            t: float(np.mean(scores)) for t, scores in sorted(type_scores.items())
        }

        valid_scores = [s for s in alignment_scores if s is not None]
        results_dict = {
            "avg_query_time": float(np.mean(query_times)) if query_times else 0,
            "query_times": query_times,
            "misaligned_indices": misaligned_indices,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "metric": "longmemeval_judge",
            "avg_alignment_score": float(np.mean(valid_scores)) if valid_scores else 0,
            "alignment_score_scores": alignment_scores,
            "judge_model": self.model,
            "ensemble_k": 1,
            "type_accuracy": type_accuracy,
            "type_counts": {t: len(scores) for t, scores in sorted(type_scores.items())},
        }

        self._print_summary(valid_scores, results_dict, misaligned_indices,
                            total_input_tokens, total_output_tokens, type_accuracy)
        return results_dict

    @staticmethod
    def _print_summary(alignment_scores, results_dict, misaligned_indices,
                       total_input_tokens, total_output_tokens, type_accuracy):
        print("\n\033[92m===== LongMemEval Ask Benchmark Results =====\033[0m")
        print(f"Number of queries evaluated: {len(alignment_scores)}")
        if alignment_scores:
            print(f"Overall Accuracy: {np.mean(alignment_scores):.2%}")
        print(f"Misaligned queries: {len(misaligned_indices)}")
        print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")
        print(f"Total Judge Tokens: {total_input_tokens + total_output_tokens}")
        if type_accuracy:
            print("\n\033[93mPer-Type Accuracy:\033[0m")
            for qtype, acc in type_accuracy.items():
                count = results_dict["type_counts"][qtype]
                print(f"  {qtype}: {acc:.2%} ({count} questions)")


class QueryExactMatchAskCalculator:
    """Ask metrics calculator scoring the database queries the agent issued.

    Each result's ``searches`` and ``aggregations`` are canonicalized and
    compared with the query's ``ground_truth_queries`` (see
    :mod:`query_exact_match`). A query scores 1 only when the issued query set
    equals the ground-truth set exactly. Unit-level recall and precision, and
    per-operator-class accuracy (as ``type_accuracy``), are reported alongside.

    Args:
        collection_aliases: Maps Weaviate collection names back to dataset
            collection names (e.g. ``{"SurfBeaches_Default": "SurfBeaches"}``).
        lm_judge: Optional LM judge calculator. When given, its alignment score
            on the final answer is computed and reported in addition.
    """

    def __init__(
        self,
        collection_aliases: Optional[dict[str, str]] = None,
        lm_judge: Optional[LMJudgeAskCalculator] = None,
    ):
        self.collection_aliases = collection_aliases or {}
        self.lm_judge = lm_judge

    def compute(self, results: list[AskResult]) -> dict:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with query exact match...\033[0m")

        scores: list[Optional[int]] = []
        recalls = []
        precisions = []
        query_times = []
        misaligned_indices = []
        details: list[Optional[dict]] = []
        class_scores: dict[str, list[int]] = {}
        phrasing_scores: dict[str, list[int]] = {}
        unreported = 0

        for i, result in enumerate(results):
            if result.query.ground_truth_queries is None:
                raise ValueError(
                    f"Query {i} has no ground_truth_queries; query exact match "
                    "requires a dataset with exact ground-truth queries (e.g. QueryAgentGym)."
                )
            if result.system_answer.startswith("[ERROR]"):
                scores.append(None)
                details.append(None)
                continue
            if result.searches is None and result.aggregations is None:
                unreported += 1

            ground_truth = canonicalize_ground_truth(result.query.ground_truth_queries)
            predicted = canonicalize_predicted(
                result.searches, result.aggregations, self.collection_aliases,
            )
            match = score_query_units(ground_truth, predicted)
            score = 1 if match["exact_match"] else 0

            scores.append(score)
            recalls.append(match["recall"])
            precisions.append(match["precision"])
            query_times.append(result.time_taken)
            if not score:
                misaligned_indices.append(i)

            metadata = result.query.metadata or {}
            for operator_class in metadata.get("operator_classes", []):
                class_scores.setdefault(operator_class, []).append(score)
            if metadata.get("phrasing"):
                phrasing_scores.setdefault(metadata["phrasing"], []).append(score)

            details.append({
                "query_exact_match": score,
                "sub_query_recall": match["recall"],
                "sub_query_precision": match["precision"],
                "ground_truth_queries": [render_unit(u) for u in ground_truth],
                "predicted_queries": [render_unit(u) for u in predicted],
                "missing_queries": match["missing"],
                "extra_queries": match["extra"],
            })

        if unreported:
            print(f"\033[93mWarning: {unreported} results carry no searches/aggregations; "
                  "this agent may not report the queries it issues.\033[0m")

        valid_scores = [s for s in scores if s is not None]
        results_dict = {
            "avg_query_time": float(np.mean(query_times)) if query_times else 0,
            "query_times": query_times,
            "misaligned_indices": misaligned_indices,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "metric": "query_exact_match",
            "avg_query_exact_match": float(np.mean(valid_scores)) if valid_scores else 0,
            "query_exact_match_scores": scores,
            "avg_sub_query_recall": float(np.mean(recalls)) if recalls else 0,
            "avg_sub_query_precision": float(np.mean(precisions)) if precisions else 0,
            "type_accuracy": {c: float(np.mean(v)) for c, v in sorted(class_scores.items())},
            "type_counts": {c: len(v) for c, v in sorted(class_scores.items())},
            "phrasing_accuracy": {p: float(np.mean(v)) for p, v in sorted(phrasing_scores.items())},
            "query_match_details": details,
        }

        self._print_summary(results_dict, len(valid_scores))

        if self.lm_judge is not None:
            judge_metrics = self.lm_judge.compute(results)
            for key in (
                "avg_alignment_score", "alignment_score_scores", "judge_model",
                "ensemble_k", "judge_reasonings", "total_input_tokens", "total_output_tokens",
            ):
                if key in judge_metrics:
                    results_dict[key] = judge_metrics[key]

        return results_dict

    @staticmethod
    def _print_summary(results_dict: dict, num_evaluated: int) -> None:
        print("\n\033[92m===== Query Exact Match Results =====\033[0m")
        print(f"Number of queries evaluated: {num_evaluated}")
        print(f"Query Exact Match: {results_dict['avg_query_exact_match']:.2%}")
        print(f"Sub-query Recall: {results_dict['avg_sub_query_recall']:.2%}")
        print(f"Sub-query Precision: {results_dict['avg_sub_query_precision']:.2%}")
        print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")
        if results_dict["type_accuracy"]:
            print("\n\033[93mPer-Operator-Class Exact Match:\033[0m")
            for operator_class, acc in results_dict["type_accuracy"].items():
                count = results_dict["type_counts"][operator_class]
                print(f"  {operator_class}: {acc:.2%} ({count} queries)")
