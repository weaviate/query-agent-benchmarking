def calculate_recall(target_id: int, retrieved_ids: list[int]):
    print(f"\033[96mTarget ID: {target_id}\033[0m")
    if target_id in retrieved_ids:
        print(f"\033[92mRetrieved IDs: {retrieved_ids}\033[0m")
        return 1
    print(f"\033[91mRetrieved IDs: {retrieved_ids}\033[0m")
    return 0