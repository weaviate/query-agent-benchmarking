"""
Data preparation utilities for DSPy optimization.

This module provides functions for loading datasets and converting them
to DSPy Examples for use in optimization pipelines.
"""

from typing import List, Dict, Tuple

from dspy import Example



def create_dspy_examples_from_dataset(
    queries: List[Dict],
    max_train: int,
    max_test: int,
    train_ratio: float = 0.8
):
    """
    Convert dataset queries to DSPy Examples.
    
    Args:
        queries: List of query dictionaries from dataset loader
        max_train: Maximum number of training examples to use
        max_test: Maximum number of test examples to use
        dataset_name: Name of the dataset for context
        
    Returns:
        Tuple of (train_examples, test_examples)
    """
    examples = []
    
    for query in queries:
        example = Example()
        example = example.with_inputs("question")
        
        example["question"] = query["question"]
        
        # Add dataset_ids for recall evaluation
        if "dataset_ids" in query:
            example.dataset_ids = query["dataset_ids"]
        
        # Add nugget data if available (for FreshStack datasets)
        if "nugget_data" in query:
            example.nugget_data = query["nugget_data"]
    
        examples.append(example)

    # Split into train/test sets (80/20 split)
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    # Apply max size limits if specified
    if max_train:
        train_examples = train_examples[:max_train]
    if max_test:
        test_examples = test_examples[:max_test]
    
    return train_examples, test_examples


def get_collection_info(dataset_name: str) -> Tuple[str, str]:
    """
    Get collection name and target property name for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (collection_name, target_property_name)
        
    Raises:
        ValueError: If dataset is not recognized
    """
    if dataset_name == "wixqa":
        collection_name = "WixKB"
        target_property_name = "contents"
    elif dataset_name == "enron":
        collection_name = "EnronEmails"
        target_property_name = ""
    elif dataset_name.startswith("freshstack-"):
        subset = dataset_name.split("-")[1].capitalize()
        collection_name = f"FreshStack{subset}"
        target_property_name = "docs_text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return collection_name, target_property_name