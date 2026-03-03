#!wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

import json

# ANSI escape codes for green and white
GREEN = "\033[92m"
WHITE = "\033[97m"
RESET = "\033[0m"

with open("longmemeval_s_cleaned.json", "r", encoding="utf-8") as f:
    longmemeval_data = json.load(f)

first_item = longmemeval_data[0]
for key, value in first_item.items():
    value_type = type(value).__name__
    print(f"{GREEN}{key}{RESET}: {WHITE}{value_type}{RESET}")

'''
question_id: str
question_type: str
question: str
question_date: str
answer: str
answer_session_ids: list
haystack_dates: list
haystack_session_ids: list
haystack_sessions: list
'''

# Create two datasets from this
'''
  # longmemeval-s-docs (one entry per session per question)
  {
    "tenant_id": "<question_id>",
    "session_id": "<haystack_session_id>",
    "session_date": "<haystack_date>",
    "session_text": "<concatenated conversation>"
  }

  # longmemeval-s-queries
  {
    "question_id": "<question_id>",
    "question": "<question>",
    "question_date": "<question_date>",
    "answer": "<answer>",
    "answer_session_ids": ["<session_id>", ...],
    "tenant_id": "<question_id>"
  }
'''

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
with open("longmemeval-s-docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2, ensure_ascii=False)

with open("longmemeval-s-queries.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2, ensure_ascii=False)

print(f"\n{GREEN}Saved {len(docs)} docs{RESET} across {len(queries)} questions")
print(f"{GREEN}  -> longmemeval-s-docs.json{RESET}")
print(f"{GREEN}  -> longmemeval-s-queries.json{RESET}")