#!wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
#!wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json

import json

# ANSI escape codes for green and white
GREEN = "\033[92m"
WHITE = "\033[97m"
RESET = "\033[0m"

def analyze_first_item(data, subset_name):
    print(f"{GREEN}Analyzing {subset_name}{RESET}")
    first_item = data[0]
    for key, value in first_item.items():
        value_type = type(value).__name__
        print(f"{GREEN}{key}{RESET}: {WHITE}{value_type}{RESET}")

def build_docs_and_queries(longmemeval_data, subset_tag):
    # Build docs: one entry per session per question
    docs = []
    for item in longmemeval_data:
        question_id = item["question_id"]
        for session, session_id, session_date in zip(
            item["haystack_sessions"],
            item["haystack_session_ids"],
            item["haystack_dates"],
        ):
            # Concatenate the multi-turn conversation into a single string
            session_text = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in session
            )
            docs.append({
                "tenant_id": question_id,
                "session_id": session_id,
                "session_date": session_date,
                "session_text": session_text,
            })

    # Build queries: one entry per question
    queries = []
    for item in longmemeval_data:
        queries.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "question_date": item["question_date"],
            "answer": str(item["answer"]),
            "answer_session_ids": item["answer_session_ids"],
            "tenant_id": item["question_id"],
        })

    # Save
    docs_filename = f"longmemeval-{subset_tag}-docs.json"
    queries_filename = f"longmemeval-{subset_tag}-queries.json"
    with open(docs_filename, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    with open(queries_filename, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    print(f"\n{GREEN}Saved {len(docs)} docs{RESET} across {len(queries)} questions")
    print(f"{GREEN}  -> {docs_filename}{RESET}")
    print(f"{GREEN}  -> {queries_filename}{RESET}")

# For both s (small) and m (medium) splits
for subset_file, subset_tag in [
    ("longmemeval_s_cleaned.json", "s"),
    ("longmemeval_m_cleaned.json", "m"),
]:
    try:
        with open(subset_file, "r", encoding="utf-8") as f:
            longmemeval_data = json.load(f)
        analyze_first_item(longmemeval_data, subset_tag)
        build_docs_and_queries(longmemeval_data, subset_tag)
    except FileNotFoundError:
        print(f"{WHITE}File not found:{RESET} {subset_file}")
