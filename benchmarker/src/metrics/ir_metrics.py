def calculate_recall(target_ids: list[str], retrieved_ids: list[str]):
    """Calculate recall for retrieved documents.
    
    Args:
        target_ids: List of target document IDs (ground truth)
        retrieved_ids: List of retrieved document IDs
        
    Returns:
        float: Recall score (0.0 to 1.0)
    """
    if not isinstance(target_ids, list):
        target_ids = [target_ids]
    
    target_ids = [str(id) for id in target_ids]
    retrieved_ids = [str(id) for id in retrieved_ids] if retrieved_ids else []
    
    print(f"\033[96mTarget IDs: {target_ids}\033[0m")
    found_count = sum(1 for target_id in target_ids if target_id in retrieved_ids)
    if found_count > 0:
        print(f"\033[92mRetrieved IDs: {retrieved_ids}\033[0m")
        return found_count / len(target_ids)
    print(f"\033[91mRetrieved IDs: {retrieved_ids}\033[0m")
    return 0