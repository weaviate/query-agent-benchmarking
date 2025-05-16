from pydantic import BaseModel
from typing import Optional
import json
import random

from datasets import load_dataset

def in_memory_dataset_loader(dataset_name: str):
    if dataset_name == "enron":
        emails = _load_dataset_from_hf_hub("weaviate/enron-qa-emails-dasovich-j")
        questions = _load_dataset_from_hf_hub("weaviate/enron-qa-questions-dasovich-j")
        # Rename dataset_id to dataset_ids for enron questions and ensure it's a list
        for question in questions:
            dataset_id = question.pop('dataset_id')
            question['dataset_ids'] = [dataset_id] if not isinstance(dataset_id, list) else dataset_id
        return emails, questions
    if dataset_name == "wixqa":
        documents = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wix_kb_corpus")
        questions = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wixqa_expertwritten")
        # Rename article_ids to dataset_ids for wixqa questions and ensure it's a list
        for question in questions:
            article_ids = question.pop('article_ids')
            question['dataset_ids'] = [article_ids] if not isinstance(article_ids, list) else article_ids
        return documents, questions
    else:
        return None

def _load_dataset_from_hf_hub(filepath, subset=None):
    ds = load_dataset(filepath, subset)
    train_dataset = ds["train"]
    
    dataset_dicts = []
    for item in train_dataset:
        dataset_dicts.append(dict(item))
    
    return dataset_dicts

def _load_dataset_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def split_dataset(dataset, train_ratio=0.8, shuffle=True):
    if shuffle:
        dataset = dataset.copy()
        random.shuffle(dataset)
        
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    return train_data, test_data