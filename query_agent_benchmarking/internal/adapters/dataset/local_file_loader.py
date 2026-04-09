"""Dataset loaders for local files (OfficeQA PDFs and CSV)."""

import csv
from pathlib import Path

from query_agent_benchmarking.internal.core.models import InMemoryQuery, InMemoryAskQuery
from query_agent_benchmarking.database.pdf_utils import get_pdf_page_counts, process_single_pdf


def load_officeqa():
    """Load OfficeQA dataset from local PDF files and CSV queries.

    Returns:
        Tuple of (docs_list, queries_list).

    Raises:
        FileNotFoundError: If the PDF directory or CSV file is missing.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    import multiprocessing
    import os

    local_data_dir = Path(__file__).parent.parent.parent.parent / "local-data"
    pdf_dir = local_data_dir / "treasury_bulletin_pdfs"
    csv_path = local_data_dir / "officeqa_pro.csv"

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print("Loading OfficeQA dataset from local PDFs...")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files, counting pages...")

    page_counts = get_pdf_page_counts(pdf_files)
    total_pages = sum(page_counts.values())
    num_workers = min(os.cpu_count() or 4, len(pdf_files))
    print(f"Total pages: {total_pages}, processing with {num_workers} processes")

    docs = []
    ctx = multiprocessing.get_context("fork")
    with tqdm(total=len(pdf_files), desc="Rendering PDFs", unit="pdf") as pbar:
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = {executor.submit(process_single_pdf, p): p for p in pdf_files}
            for future in as_completed(futures):
                pages = future.result()
                docs.extend(pages)
                pbar.update(1)
                pbar.set_postfix(pages=len(docs))

    docs.sort(key=lambda d: (d["source_pdf"], d["page_number"]))
    print(f"Converted {len(docs)} PDF pages to base64 images")

    questions: list[InMemoryQuery] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(
                InMemoryQuery(
                    question=row["question"],
                    query_id=row["uid"],
                    dataset_ids=[],
                )
            )

    print(f"Loaded {len(docs)} documents and {len(questions)} questions")
    return docs, questions


def load_ask_officeqa() -> list[InMemoryAskQuery]:
    """Load OfficeQA ask queries from local CSV.

    Returns:
        List of InMemoryAskQuery instances.

    Raises:
        FileNotFoundError: If the CSV file is missing.
    """
    csv_path = Path(__file__).parent.parent.parent.parent / "local-data" / "officeqa_pro.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print("Loading OfficeQA ask queries...")
    queries: list[InMemoryAskQuery] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(
                InMemoryAskQuery(
                    question=row["question"],
                    ground_truth_answer=row["answer"],
                    oracle_context_id=None,
                )
            )

    print(f"Loaded {len(queries)} ask queries")
    return queries
