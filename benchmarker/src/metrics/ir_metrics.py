def calculate_recall(target_ids: list[str], retrieved_ids: list[str], nugget_data: list[dict] = None):
    """Calculate recall for retrieved documents.
    
    Args:
        target_ids: List of target document IDs (ground truth)
        retrieved_ids: List of retrieved document IDs
        nugget_data: Optional list of nugget information for nugget-based evaluation
                    Each nugget should have 'relevant_corpus_ids' field
        
    Returns:
        float: Recall score (0.0 to 1.0)
               For nugget-based evaluation: percentage of nuggets with at least one relevant doc retrieved
               For standard evaluation: traditional recall (found relevant / total relevant)
    """
    if not isinstance(target_ids, list):
        target_ids = [target_ids]
    
    target_ids = [str(id) for id in target_ids]
    retrieved_ids = [str(id) for id in retrieved_ids] if retrieved_ids else []
    
    # If nugget_data is provided, use nugget-based evaluation
    if nugget_data is not None:
        print(f"\033[96mUsing nugget-based evaluation with {len(nugget_data)} nuggets\033[0m")
        
        nuggets_with_hits = 0
        for i, nugget in enumerate(nugget_data):
            nugget_relevant_ids = [str(id) for id in nugget['relevant_corpus_ids']]
            nugget_hits = [id for id in nugget_relevant_ids if id in retrieved_ids]
            
            if nugget_hits:
                nuggets_with_hits += 1
                print(f"\033[92mNugget {i+1}: Found {len(nugget_hits)} relevant docs: {nugget_hits}\033[0m")
            else:
                print(f"\033[91mNugget {i+1}: No relevant docs found. Expected: {nugget_relevant_ids}\033[0m")
        
        nugget_recall = nuggets_with_hits / len(nugget_data) if nugget_data else 0
        print(f"\033[96mNugget recall: {nuggets_with_hits}/{len(nugget_data)} = {nugget_recall:.2f}\033[0m")
        return nugget_recall
    
    # Standard recall evaluation
    print(f"\033[96mTarget IDs: {target_ids}\033[0m")
    found_count = sum(1 for target_id in target_ids if target_id in retrieved_ids)
    if found_count > 0:
        print(f"\033[92mRetrieved IDs: {retrieved_ids}\033[0m")
        return found_count / len(target_ids)
    print(f"\033[91mRetrieved IDs: {retrieved_ids}\033[0m")
    return 0