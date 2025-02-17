from pydantic import BaseModel
from typing import Optional
import json
import random

from benchmarker.cmd.models.weaviate_query import WeaviateQuery

class QueryAgentTest(BaseModel):
    natural_language_command: str
    database_schema: str
    gold_answer: Optional[str]
    ground_truth_queries: Optional[list[WeaviateQuery]] = None

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        return [QueryAgentTest(**item) for item in data]

def split_dataset(dataset, train_ratio=0.8, shuffle=True):
    if shuffle:
        dataset = dataset.copy()
        random.shuffle(dataset)
        
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    return train_data, test_data