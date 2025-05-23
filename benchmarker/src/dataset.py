import json
import random

from datasets import load_dataset

def in_memory_dataset_loader(dataset_name: str):
    if dataset_name == "enron":
        return _in_memory_dataset_loader_enron()
    elif dataset_name == "wixqa":
        return _in_memory_dataset_loader_wixqa()
    elif dataset_name == "freshstack-langchain":
        return _in_memory_dataset_loader_freshstackqa_langchain()
    else:
        return None

def _in_memory_dataset_loader_enron():
    emails = _load_dataset_from_hf_hub("weaviate/enron-qa-emails-dasovich-j")
    questions = _load_dataset_from_hf_hub("weaviate/enron-qa-questions-dasovich-j")
    for question in questions:
        dataset_id = question.pop('dataset_id')
        question['dataset_ids'] = [dataset_id] if not isinstance(dataset_id, list) else dataset_id
    return emails, questions

def _in_memory_dataset_loader_wixqa():
    documents = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wix_kb_corpus")
    questions = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wixqa_expertwritten")
    for question in questions:
        article_ids = question.pop('article_ids')
        question['dataset_ids'] = [article_ids] if not isinstance(article_ids, list) else article_ids
    return documents, questions

def _in_memory_dataset_loader_freshstackqa_langchain():
    docs = _load_dataset_from_hf_hub(filepath="freshstack/corpus-oct-2024", subset="langchain")
    for doc in docs:
        doc['dataset_id'] = doc.pop('_id')
    questions = _load_dataset_from_hf_hub(
        filepath="freshstack/queries-oct-2024", 
        subset="langchain", 
        train=False
    )

    for question in questions:
        all_relevant_ids = []
        nugget_data = []
        ids_per_nugget = {}
        
        for i, nugget in enumerate(question.get('nuggets', [])):
            nugget_id = f"{question['query_id']}_nugget_{i}"
            nugget_text = nugget['text']
            relevant_corpus_ids = nugget['relevant_corpus_ids']
            
            nugget_info = {
                'nugget_id': nugget_id,
                'text': nugget_text,
                'relevant_corpus_ids': relevant_corpus_ids
            }
            nugget_data.append(nugget_info)
            all_relevant_ids.extend(relevant_corpus_ids)
            
            ids_per_nugget[nugget_text] = relevant_corpus_ids
        
        unique_relevant_ids = list(dict.fromkeys(all_relevant_ids))
        
        question['dataset_ids'] = unique_relevant_ids
        question['ids_per_nugget'] = ids_per_nugget
        question['nugget_data'] = nugget_data
        question['num_nuggets'] = len(nugget_data)
        question["question"] = question["query_text"]
    
    return docs, questions

def _load_dataset_from_hf_hub(filepath, subset=None, train=True):
    ds = load_dataset(filepath, subset)
    if train:
        train_dataset = ds["train"]
    else:
        train_dataset = ds["test"]
    
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