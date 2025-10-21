import json
import random
import os
import weaviate
from datasets import load_dataset

from query_agent_benchmarking.models import InMemoryQuery

def in_memory_dataset_loader(dataset_name: str):
    if dataset_name == "enron":
        return _in_memory_dataset_loader_enron()
    elif dataset_name == "wixqa":
        return _in_memory_dataset_loader_wixqa()
    elif dataset_name.startswith("beir/"):
        return _in_memory_dataset_loader_beir(dataset_name)
    elif dataset_name.startswith("bright/"):
        return _in_memory_dataset_loader_bright(dataset_name)
    elif dataset_name.startswith("lotte/"):
        return _in_memory_dataset_loader_lotte(dataset_name)
    elif dataset_name == "freshstack-angular":
        return _in_memory_dataset_loader_freshstack(subset="angular")
    elif dataset_name == "freshstack-godot":
        return _in_memory_dataset_loader_freshstack(subset="godot")
    elif dataset_name == "freshstack-langchain":
        return _in_memory_dataset_loader_freshstack(subset="langchain")
    elif dataset_name == "freshstack-laravel":
        return _in_memory_dataset_loader_freshstack(subset="laravel")
    elif dataset_name == "freshstack-yolo":
        return _in_memory_dataset_loader_freshstack(subset="yolo")
    else:
        return None

def _in_memory_dataset_loader_beir(dataset_name: str):
    import ir_datasets
    dataset = ir_datasets.load(f"{dataset_name}")
    print(f"Loading BEIR dataset: {dataset_name}")
    docs, questions = [], []
    for doc in dataset.docs_iter():
        docs.append({
        "title": getattr(doc, "title", ""),
        "content": getattr(doc, "text", ""),
        "doc_id": getattr(doc, "doc_id", None)
    })
    qrels = {}
    for qrel in dataset.qrels_iter():
        query_id = qrel.query_id
        if query_id not in qrels:
            qrels[query_id] = []
        qrels[query_id].append(qrel.doc_id)
    for question in dataset.queries_iter():
        questions.append({
            InMemoryQuery(
                question=question.text,
                query_id=question.query_id,
                dataset_ids=qrels[question.query_id]
            )
        })
    return docs, questions

def _in_memory_dataset_loader_bright(dataset_name: str):
    all_docs = load_dataset("xlangai/BRIGHT", "documents")
    split = dataset_name.split("/")[1]
    print(f"Loading BRIGHT dataset: {dataset_name}")
    docs, questions = [], []
    for doc in all_docs[split]:
        docs.append({
            "content": doc["content"],
            "dataset_id": doc["id"]
        })
    all_questions = load_dataset("xlangai/BRIGHT", "examples")
    for question in all_questions[split]:
        questions.append(
            InMemoryQuery(
                question=question["query"],
                query_id=question["id"],
                dataset_ids=question["gold_ids"]
            )
        )
    return docs, questions

def _in_memory_dataset_loader_lotte(dataset_name: str):
    import ir_datasets
    dataset = ir_datasets.load(f"{dataset_name}")
    print(f"Loading LOTTE dataset: {dataset_name}")
    docs, questions = [], []
    for doc in dataset.docs_iter():
        docs.append({
        "text": getattr(doc, "text", ""),
        "doc_id": getattr(doc, "doc_id", None)
    })
    qrels = {}
    for qrel in dataset.qrels_iter():
        query_id = qrel.query_id
        if query_id not in qrels:
            qrels[query_id] = []
        qrels[query_id].append(qrel.doc_id)
    for question in dataset.queries_iter():
        questions.append(
            InMemoryQuery(
                question=question.text,
                query_id=question.query_id,
                dataset_ids=qrels[question.query_id]
            )
        )
    return docs, questions

def _in_memory_dataset_loader_enron():
    emails = _load_dataset_from_hf_hub("weaviate/enron-qa-emails-dasovich-j")
    questions = _load_dataset_from_hf_hub("weaviate/enron-qa-questions-dasovich-j")
    for question in questions:
        dataset_ids = question.pop('dataset_id')
        # Need to convert these to strings
        questions.append(
            InMemoryQuery(
                question=question["question"],
                query_id=question["query_id"],
                dataset_ids=[dataset_ids] if not isinstance(dataset_ids, list) else dataset_ids
            )
        )
    return emails, questions

def _in_memory_dataset_loader_wixqa():
    documents = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wix_kb_corpus")
    questions = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wixqa_expertwritten")
    for question in questions:
        article_ids = question.pop('article_ids')
        questions.append(InMemoryQuery(
            question=question["question"],
            query_id=question["query_id"],
            dataset_ids=[article_ids] if not isinstance(article_ids, list) else article_ids
        ))
    return documents, questions

# Need to check how this benchmark needs to extend the `InMemoryQuery` model
def _in_memory_dataset_loader_freshstack(subset: str):
    pass
    """
    docs = _load_dataset_from_hf_hub(filepath="freshstack/corpus-oct-2024", subset=subset)
    for doc in docs:
        doc['dataset_id'] = doc.pop('_id')
    questions = _load_dataset_from_hf_hub(
        filepath="freshstack/queries-oct-2024", 
        subset=subset, 
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
    """

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

# Update me!
# TODO: Update to just take a sample of the queries
def load_queries_from_weaviate_collection(
    collection_name: str, 
    query_content_key: str, 
    gold_ids_key: str
):
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
    )

    query_collection = weaviate_client.collections.get(collection_name)
    
    queries: list[InMemoryQuery] = []

    for query_item in query_collection.iterator():
        props = query_item.properties
        query = InMemoryQuery(
            question=props[query_content_key],
            dataset_ids=props[gold_ids_key]
        )
        queries.append(query)
    return queries



def split_dataset(dataset, train_ratio=0.8, shuffle=True):
    if shuffle:
        dataset = dataset.copy()
        random.shuffle(dataset)
        
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    return train_data, test_data