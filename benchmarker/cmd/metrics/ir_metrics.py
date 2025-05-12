def calculate_recall_at_k(target_id, retrieved_ids, k):
    if target_id in retrieved_ids[:k]:
        return 1
    return 0