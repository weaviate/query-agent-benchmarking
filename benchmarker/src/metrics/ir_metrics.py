import numpy as np

def calculate_recall(target_ids: list[str], retrieved_ids: list[str]):
    """Calculate traditional recall for retrieved documents.
    
    Args:
        target_ids: List of target document IDs (ground truth)
        retrieved_ids: List of retrieved document IDs
        
    Returns:
        float: Recall score (0.0 to 1.0) - proportion of relevant docs retrieved
    """
    if not isinstance(target_ids, list):
        target_ids = [target_ids]
    
    target_ids = [str(id) for id in target_ids]
    retrieved_ids = [str(id) for id in retrieved_ids] if retrieved_ids else []
    
    print(f"\033[96mTarget IDs: {target_ids}\033[0m")
    found_count = sum(1 for target_id in target_ids if target_id in retrieved_ids)
    
    if found_count > 0:
        print(f"\033[92mRetrieved IDs: {retrieved_ids}\033[0m")
    else:
        print(f"\033[91mRetrieved IDs: {retrieved_ids}\033[0m")
    
    recall = found_count / len(target_ids) if target_ids else 0
    print(f"\033[96mRecall: {found_count}/{len(target_ids)} = {recall:.2f}\033[0m")
    
    return recall


def calculate_coverage(retrieved_ids: list[str], nugget_data: list[dict], k: int = 20):
    """Calculate Coverage@k metric from FreshStack.
    
    Measures the proportion of nuggets covered by the top-k retrieved documents.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        nugget_data: List of nugget information, each with 'relevant_corpus_ids' field
        k: Number of top documents to consider (default: 20)
    
    Returns:
        float: Coverage@k score (0.0 to 1.0) - proportion of nuggets covered
    """
    if not nugget_data:
        return 0.0
    
    # Convert to strings for consistent comparison
    retrieved_ids = [str(id) for id in retrieved_ids[:k]] if retrieved_ids else []
    
    covered_nuggets = set()
    nugget_coverage_details = []
    
    for i, nugget in enumerate(nugget_data):
        nugget_id = nugget.get('id', f'nugget_{i}')
        nugget_relevant_ids = [str(id) for id in nugget.get('relevant_corpus_ids', [])]
        
        # Check if any relevant doc for this nugget is in top-k retrieved
        covered = any(doc_id in retrieved_ids for doc_id in nugget_relevant_ids)
        
        if covered:
            covered_nuggets.add(nugget_id)
            nugget_coverage_details.append(f"\033[92mNugget {i+1}: Covered\033[0m")
        else:
            nugget_coverage_details.append(f"\033[91mNugget {i+1}: Not covered\033[0m")
    
    coverage_score = len(covered_nuggets) / len(nugget_data)
    
    # Print summary
    print(f"\033[96mCoverage@{k} evaluation:\033[0m")
    print(f"Total nuggets: {len(nugget_data)}")
    print(f"Covered nuggets: {len(covered_nuggets)}")
    for detail in nugget_coverage_details[:5]:  # Show first 5 for brevity
        print(detail)
    if len(nugget_coverage_details) > 5:
        print(f"... and {len(nugget_coverage_details) - 5} more nuggets")
    print(f"\033[96mCoverage@{k}: {len(covered_nuggets)}/{len(nugget_data)} = {coverage_score:.2f}\033[0m")
    
    return coverage_score


def calculate_alpha_ndcg(retrieved_ids: list[str], nugget_data: list[dict], alpha: float = 0.5, k: int = 10):
    """Calculate α-nDCG@k for diversity-aware ranking.
    
    This metric rewards both relevance and diversity, penalizing redundant coverage
    of the same nuggets.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        nugget_data: List of nugget information, each with 'relevant_corpus_ids' field
        alpha: Penalization factor for redundancy (0-1). 
               0 = maximum penalty for redundancy, 1 = no penalty
        k: Cutoff for evaluation (default: 10)
    
    Returns:
        float: α-nDCG@k score (0.0 to 1.0)
    """
    if not nugget_data or not retrieved_ids:
        return 0.0
    
    # Convert to strings for consistent comparison
    retrieved_ids = [str(id) for id in retrieved_ids[:k]]
    
    # Track which nuggets have been covered
    covered_nuggets = set()
    dcg = 0.0
    position_gains = []
    
    # Calculate DCG
    for i, doc_id in enumerate(retrieved_ids):
        position = i + 1
        doc_gain = 0.0
        doc_nuggets = []
        
        # Find nuggets this document covers
        for j, nugget in enumerate(nugget_data):
            nugget_id = nugget.get('id', f'nugget_{j}')
            nugget_relevant_ids = [str(id) for id in nugget.get('relevant_corpus_ids', [])]
            
            if doc_id in nugget_relevant_ids:
                if nugget_id not in covered_nuggets:
                    # First time covering this nugget - full credit
                    doc_gain += 1.0
                    covered_nuggets.add(nugget_id)
                    doc_nuggets.append(f"new:{nugget_id}")
                else:
                    # Redundant coverage - penalized by (1-α)
                    doc_gain += (1 - alpha)
                    doc_nuggets.append(f"redundant:{nugget_id}")
        
        # Apply position discount
        position_discount = 1.0 / np.log2(position + 1)
        position_contribution = doc_gain * position_discount
        dcg += position_contribution
        
        position_gains.append({
            'position': position,
            'doc_id': doc_id,
            'gain': doc_gain,
            'discount': position_discount,
            'contribution': position_contribution,
            'nuggets': doc_nuggets
        })
    
    # Calculate ideal DCG
    # Ideal: each position covers a new nugget until all are covered
    idcg = 0.0
    for i in range(min(len(nugget_data), k)):
        position = i + 1
        ideal_gain = 1.0  # Each position ideally covers a new nugget
        idcg += ideal_gain / np.log2(position + 1)
    
    # Calculate α-nDCG
    alpha_ndcg = dcg / idcg if idcg > 0 else 0.0
    
    # Print detailed breakdown
    print(f"\033[96mα-nDCG@{k} evaluation (α={alpha}):\033[0m")
    print(f"Total nuggets: {len(nugget_data)}")
    print(f"Documents evaluated: {len(retrieved_ids)}")
    print(f"\nPosition breakdown:")
    for pg in position_gains[:5]:  # Show first 5 positions
        nugget_info = ', '.join(pg['nuggets']) if pg['nuggets'] else 'none'
        print(f"  Pos {pg['position']}: doc={pg['doc_id'][:8]}... gain={pg['gain']:.2f} "
              f"discount={pg['discount']:.3f} contrib={pg['contribution']:.3f} nuggets=[{nugget_info}]")
    if len(position_gains) > 5:
        print(f"  ... and {len(position_gains) - 5} more positions")
    
    print(f"\nDCG: {dcg:.3f}, IDCG: {idcg:.3f}")
    print(f"\033[96mα-nDCG@{k}: {alpha_ndcg:.3f}\033[0m")
    
    return alpha_ndcg