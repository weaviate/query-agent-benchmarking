"""Dataset loaders using the ir_datasets library (BEIR, LOTTE)."""

from query_agent_benchmarking.internal.core.domain.models import InMemoryQuery


def load_beir(dataset_name: str):
    """Load a BEIR dataset via ir_datasets.

    Args:
        dataset_name: Full BEIR dataset path (e.g., "beir/scifact/test").

    Returns:
        Tuple of (docs_list, queries_list).
    """
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    print(f"Loading BEIR dataset: {dataset_name}")

    docs = []
    for doc in dataset.docs_iter():
        docs.append({
            "title": getattr(doc, "title", ""),
            "content": getattr(doc, "text", ""),
            "doc_id": getattr(doc, "doc_id", None),
        })

    qrels: dict[str, list[str]] = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, []).append(qrel.doc_id)

    questions = []
    for question in dataset.queries_iter():
        questions.append(
            InMemoryQuery(
                question=question.text,
                query_id=question.query_id,
                dataset_ids=qrels[question.query_id],
            )
        )

    return docs, questions


def load_lotte(dataset_name: str):
    """Load a LOTTE dataset via ir_datasets.

    Args:
        dataset_name: Full LOTTE dataset path (e.g., "lotte/lifestyle/test/forum").

    Returns:
        Tuple of (docs_list, queries_list).
    """
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    print(f"Loading LOTTE dataset: {dataset_name}")

    docs = []
    for doc in dataset.docs_iter():
        docs.append({
            "text": getattr(doc, "text", ""),
            "doc_id": getattr(doc, "doc_id", None),
        })

    qrels: dict[str, list[str]] = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, []).append(qrel.doc_id)

    questions = []
    for question in dataset.queries_iter():
        questions.append(
            InMemoryQuery(
                question=question.text,
                query_id=question.query_id,
                dataset_ids=qrels[question.query_id],
            )
        )

    return docs, questions
