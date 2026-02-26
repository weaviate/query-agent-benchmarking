import json
import random
from typing import Optional

from datasets import load_dataset

from query_agent_benchmarking.utils import get_weaviate_client

from query_agent_benchmarking.models import InMemoryQuery, InMemoryAskQuery, NuggetInfo

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
    elif dataset_name.startswith("irpapers/"):
        return _in_memory_dataset_loader_irpapers(dataset_name)
    elif dataset_name == "multihoprag":
        return _in_memory_dataset_loader_multihoprag()
    elif dataset_name == "vidore_v3_hr":
        return _in_memory_dataset_loader_vidore()
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
    parsed_questions = []
    for question in questions:
        dataset_ids = question.pop('dataset_id')
        # Need to convert these to strings
        parsed_questions.append(
            InMemoryQuery(
                question=question["question"],
                query_id=question["query_id"],
                dataset_ids=[dataset_ids] if not isinstance(dataset_ids, list) else dataset_ids
            )
        )
    return emails, parsed_questions

def _in_memory_dataset_loader_wixqa():
    documents = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wix_kb_corpus")
    questions = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wixqa_expertwritten")
    parsed_questions = []
    for question in questions:
        article_ids = question.pop('article_ids')
        parsed_questions.append(InMemoryQuery(
            question=question["question"],
            query_id=question["query_id"],
            dataset_ids=[article_ids] if not isinstance(article_ids, list) else article_ids
        ))
    return documents, parsed_questions

def _in_memory_dataset_loader_freshstack(subset: str):    
    docs = _load_dataset_from_hf_hub(filepath="freshstack/corpus-oct-2024", subset=subset)
    for doc in docs:
        doc['dataset_id'] = doc.pop('_id')
    raw_questions = _load_dataset_from_hf_hub(
        filepath="freshstack/queries-oct-2024", 
        subset=subset, 
        train=False
    )

    questions = []
    for question in raw_questions:
        all_relevant_ids = []
        nugget_data = []
        ids_per_nugget = {}
        
        for i, nugget in enumerate(question.get('nuggets', [])):
            nugget_id = f"{question['query_id']}_nugget_{i}"
            nugget_text = nugget['text']
            relevant_corpus_ids = nugget['relevant_corpus_ids']
            
            nugget_info = NuggetInfo(
                nugget_id=nugget_id,
                text=nugget_text,
                relevant_corpus_ids=relevant_corpus_ids
            )
            nugget_data.append(nugget_info)
            all_relevant_ids.extend(relevant_corpus_ids)
            
            ids_per_nugget[nugget_text] = relevant_corpus_ids
        
        unique_relevant_ids = list(dict.fromkeys(all_relevant_ids))
        
        questions.append(
            InMemoryQuery(
                question=question["query_text"],
                query_id=question["query_id"],
                dataset_ids=unique_relevant_ids,
                nugget_data=nugget_data,
                ids_per_nugget=ids_per_nugget,
                num_nuggets=len(nugget_data)
            )
        )
    
    return docs, questions

def _in_memory_dataset_loader_irpapers(dataset_name: str):
    docs = _load_dataset_from_hf_hub(filepath="weaviate/irpapers-docs")

    parts = dataset_name.split("/") # e.g. `irpapers/images/visual-queries`
    query_type = parts[2] if len(parts) > 2 else ""
    if query_type == "visual-queries":
        _questions = _load_dataset_from_hf_hub(filepath="weaviate/irpapers-visual-queries")
    else:
        _questions = _load_dataset_from_hf_hub(filepath="weaviate/irpapers-queries")

    questions: list[InMemoryQuery] = []
    for question in _questions:
        questions.append(InMemoryQuery(
            question=question["question"],
            query_id=str(random.randint(1, 1000000)),
            dataset_ids=[question["dataset_id"]]
        ))

    return docs, questions

def _in_memory_dataset_loader_vidore():
    print("Loading ViDoRe dataset...")

    corpus = load_dataset("vidore/vidore_v3_hr", "corpus")["test"]
    docs = []
    for item in corpus:
        docs.append({
            "markdown": item["markdown"],
            "dataset_id": str(item["corpus_id"]),
        })

    raw_qrels = load_dataset("vidore/vidore_v3_hr", "qrels")["test"]
    qrels = {}
    for item in raw_qrels:
        if item["score"] > 0:
            query_id = str(item["query_id"])
            if query_id not in qrels:
                qrels[query_id] = []
            qrels[query_id].append(str(item["corpus_id"]))

    raw_queries = load_dataset("vidore/vidore_v3_hr", "queries")["test"]
    questions = []
    for item in raw_queries:
        query_id = str(item["query_id"])
        if query_id in qrels:
            questions.append(
                InMemoryQuery(
                    question=item["query"],
                    query_id=query_id,
                    dataset_ids=qrels[query_id],
                )
            )

    print(f"Loaded {len(docs)} documents and {len(questions)} questions")
    return docs, questions

def _in_memory_dataset_loader_multihoprag():
    print("Loading MultiHopRAG dataset...")
    
    docs = _load_dataset_from_hf_hub(filepath="yixuantt/MultiHopRAG", subset="corpus")
    _questions = _load_dataset_from_hf_hub(filepath="yixuantt/MultiHopRAG", subset="MultiHopRAG")
    
    for i, doc in enumerate(docs):
        doc["dataset_id"] = str(i)
    
    questions: list[InMemoryQuery] = []
    for question in _questions:
        questions.append(InMemoryQuery(
            question=question["query"],
            query_id=str(random.randint(1, 1000000)),
            dataset_ids=[]
        ))
    
    print(f"Loaded {len(docs)} documents and {len(questions)} questions")
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

# Update me!
# TODO: Update to just take a sample of the queries
def load_queries_from_weaviate_collection(
    collection_name: str, 
    query_content_key: str, 
    gold_ids_key: str
):
    weaviate_client = get_weaviate_client()

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


# ============================================================================
# Ask Query Loaders
# ============================================================================

def in_memory_ask_dataset_loader(dataset_name: str) -> list[InMemoryAskQuery]:
    """
    Load ask queries from a supported dataset.
    
    This abstracts away the underlying key mappings - users just specify the dataset name.
    Similar to in_memory_dataset_loader() for search benchmarks.
    
    Note: The dataset_name also determines which Weaviate collection the agent uses:
    - "irpapers/text" -> IRPapersText_Default
    - "irpapers/images" -> IRPapersImages_Default
    - "multihoprag" -> MultiHopRAG_Default
    """
    if dataset_name.startswith("irpapers/"):
        return _in_memory_ask_loader_irpapers(dataset_name)
    elif dataset_name == "multihoprag":
        return _in_memory_ask_loader_multihoprag()
    else:
        raise ValueError(
            f"Unknown ask dataset: {dataset_name}. "
            f"Supported ask datasets: irpapers/text, irpapers/images, multihoprag"
        )


def _in_memory_ask_loader_irpapers(dataset_name: str) -> list[InMemoryAskQuery]:
    """Load the IRPapers ask queries dataset."""
    print(f"Loading IRPapers ask queries for {dataset_name}...")
    
    _questions = _load_dataset_from_hf_hub(filepath="weaviate/irpapers-queries")
    
    queries: list[InMemoryAskQuery] = []
    for item in _questions:
        queries.append(InMemoryAskQuery(
            question=item["question"],
            ground_truth_answer=item["answer"],
            oracle_context_id=str(item["dataset_id"]),
        ))
    
    print(f"Loaded {len(queries)} ask queries")
    return queries


def _in_memory_ask_loader_multihoprag() -> list[InMemoryAskQuery]:
    """Load the MultiHop-RAG ask queries dataset."""
    print("Loading MultiHop-RAG ask queries...")
    
    _questions = _load_dataset_from_hf_hub(filepath="yixuantt/MultiHopRAG", subset="MultiHopRAG")
    
    queries: list[InMemoryAskQuery] = []
    for item in _questions:
        queries.append(InMemoryAskQuery(
            question=item["query"],
            ground_truth_answer=item["answer"],
            oracle_context_id=None,  # MultiHop-RAG doesn't provide oracle context IDs
        ))
    
    print(f"Loaded {len(queries)} ask queries")
    return queries

def load_ask_queries_from_weaviate(
    collection_name: str,
    query_content_key: str,
    answer_key: str,
    oracle_context_id_key: Optional[str] = None,
) -> list[InMemoryAskQuery]:
    """
    Load ask queries from a custom Weaviate collection.
    
    Use this for custom collections not in the built-in registry.
    For built-in datasets, use in_memory_ask_dataset_loader() instead.
    """
    client = get_weaviate_client()
    
    try:
        collection = client.collections.get(collection_name)
        
        return_props = [query_content_key, answer_key]
        if oracle_context_id_key:
            return_props.append(oracle_context_id_key)
        
        response = collection.query.fetch_objects(
            return_properties=return_props,
            limit=10000  # Adjust as needed
        )
        
        queries = []
        for obj in response.objects:
            oracle_id = None
            if oracle_context_id_key:
                oracle_id = str(obj.properties.get(oracle_context_id_key))
            
            queries.append(InMemoryAskQuery(
                question=obj.properties[query_content_key],
                ground_truth_answer=obj.properties[answer_key],
                oracle_context_id=oracle_id,
            ))
        
        return queries
    finally:
        client.close()