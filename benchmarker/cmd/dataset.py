from pydantic import BaseModel
from typing import Optional
import json
import random

from datasets import load_dataset

from benchmarker.cmd.models.weaviate_query import WeaviateQuery

class QueryAgentTest(BaseModel):
    natural_language_command: str
    database_schema: str
    gold_answer: Optional[str]
    ground_truth_queries: Optional[list[WeaviateQuery]] = None

def load_dataset_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        return [QueryAgentTest(**item) for item in data]

def load_dataset_from_hf_hub(filepath):
    ds = load_dataset("weaviate/enron-qa-dasovich-j")

def split_dataset(dataset, train_ratio=0.8, shuffle=True):
    if shuffle:
        dataset = dataset.copy()
        random.shuffle(dataset)
        
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    return train_data, test_data