"""
DSPy-compatible metrics for RAG evaluation.

This module provides metric functions that can be used with DSPy optimizers
to evaluate RAG programs on recall and answer quality. These functions wrap
the existing evaluation functions from the metrics module.
"""

import asyncio
from typing import Callable

from dspy import Example

from benchmarker.src.metrics.ir_metrics import calculate_recall
from benchmarker.src.metrics.lm_as_judge_agent import lm_as_judge_agent, LMJudgeAgentDeps
from benchmarker.src.utils import qa_source_parser


def get_collection(weaviate_client, dataset_name: str):
    """Get the appropriate Weaviate collection for a dataset."""
    if dataset_name == "enron":
        return weaviate_client.collections.get("EnronEmails")
    elif dataset_name == "wixqa":
        return weaviate_client.collections.get("WixKB")
    elif dataset_name.startswith("freshstack-"):
        subset = dataset_name.split("-")[1].capitalize()
        return weaviate_client.collections.get(f"FreshStack{subset}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_recall_metric(weaviate_client, dataset_name: str, weight: float = 1.0) -> Callable:
    """
    Create a recall metric function that wraps the existing calculate_recall function.
    
    Args:
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        weight: Weight to apply to the recall score
        
    Returns:
        Function that calculates recall score for a single example
    """
    collection = get_collection(weaviate_client, dataset_name)
    
    def recall_metric(example: Example, prediction, trace=None) -> float:
        try:
            # Extract sources from prediction
            if hasattr(prediction, 'sources') and prediction.sources:
                retrieved_ids = qa_source_parser(prediction.sources, collection)
            else:
                retrieved_ids = []
            
            # Get target IDs from example
            target_ids = example.dataset_ids
            
            # Use nugget-based evaluation if available
            nugget_data = getattr(example, 'nugget_data', None)
            
            # Use the existing calculate_recall function
            recall_score = calculate_recall(
                target_ids=target_ids,
                retrieved_ids=retrieved_ids,
                nugget_data=nugget_data
            )
            
            return recall_score * weight
            
        except Exception as e:
            print(f"Error calculating recall: {e}")
            return 0.0
            
    return recall_metric


def create_lm_judge_metric(weight: float = 1.0, model: str = "openai:gpt-4o") -> Callable:
    """
    Create an LM-as-a-Judge metric function that wraps the existing lm_as_judge_agent.
    
    Args:
        weight: Weight to apply to the judge score
        model: Model to use for judging
        
    Returns:
        Function that calculates LM judge score for a single example
    """
    def lm_judge_metric(example: Example, prediction, trace=None) -> float:
        try:
            # Extract answer from prediction
            if hasattr(prediction, 'final_answer'):
                answer = prediction.final_answer
            elif hasattr(prediction, 'answer'):
                answer = prediction.answer
            else:
                answer = str(prediction)
            
            # Skip if no answer provided
            if not answer or answer.strip() == "":
                return 0.0
            
            # Create dependencies for the LM judge agent
            deps = LMJudgeAgentDeps(
                question=example.question,
                system_response=answer
            )
            
            # Run async evaluation in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Use the existing lm_as_judge_agent
                result = loop.run_until_complete(
                    lm_as_judge_agent.run(deps=deps, model=model)
                )
                # Normalize from 1-5 scale to 0-1 scale
                normalized_score = (result.data.rating - 1) / 4
                return normalized_score * weight
            finally:
                loop.close()
                
        except Exception as e:
            print(f"Error calculating LM judge score: {e}")
            return 0.0
            
    return lm_judge_metric


def create_composite_metric(
    weaviate_client,
    dataset_name: str,
    recall_weight: float = 0.5,
    lm_judge_weight: float = 0.5,
    lm_judge_model: str = "openai:gpt-4o"
) -> Callable:
    """
    Create a composite metric function combining recall and LM judge metrics.
    
    Args:
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        recall_weight: Weight for recall score
        lm_judge_weight: Weight for LM judge score
        lm_judge_model: Model to use for judging
        
    Returns:
        Function that calculates composite score for a single example
    """
    recall_metric = create_recall_metric(weaviate_client, dataset_name, recall_weight)
    lm_judge_metric = create_lm_judge_metric(lm_judge_weight, lm_judge_model)
    total_weight = recall_weight + lm_judge_weight
    
    def composite_metric(example: Example, prediction, trace=None) -> float:
        recall_score = recall_metric(example, prediction, trace)
        lm_judge_score = lm_judge_metric(example, prediction, trace)
        
        composite_score = (recall_score + lm_judge_score) / total_weight
        return composite_score
        
    return composite_metric


def create_metric(
    metric_type: str,
    weaviate_client,
    dataset_name: str,
    **kwargs
) -> Callable:
    """
    Factory function for creating metric functions.
    
    Args:
        metric_type: Type of metric ("recall", "lm_judge", "composite")
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        **kwargs: Additional arguments for metric configuration
        
    Returns:
        Configured metric function
    """
    if metric_type == "recall":
        return create_recall_metric(weaviate_client, dataset_name, **kwargs)
    elif metric_type == "lm_judge":
        return create_lm_judge_metric(**kwargs)
    elif metric_type == "composite":
        return create_composite_metric(weaviate_client, dataset_name, **kwargs)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")