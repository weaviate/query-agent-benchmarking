import numpy as np

def calculate_recall_at_k(
    target_ids: list[str], 
    retrieved_ids: list[str], 
    k: int, 
    verbose: bool = False
):
    """Calculate traditional recall@k for retrieved documents.
    
    Args:
        target_ids: List of target document IDs (ground truth).
        retrieved_ids: List of retrieved document IDs.
        k: The number of top results to consider for recall calculation.
        
    Returns:
        float: Recall@k score (0.0 to 1.0) - proportion of relevant docs
               found in the top k retrieved results.
    """
    target_id_set = {str(id) for id in target_ids}

    retrieved_ids = [str(id) for id in retrieved_ids] if retrieved_ids else []

    retrieved_ids_at_k = retrieved_ids[:k]
    
    if verbose:
        print(f"\033[96mTarget IDs: {target_id_set}\033[0m")

    found_count = sum(1 for retrieved_id in retrieved_ids_at_k if retrieved_id in target_id_set)
    
    if verbose:
        print(f"\033[92mRetrieved IDs @{k}: {retrieved_ids_at_k}\033[0m")

    # Compute Success @ 1 instead of Recall @ 1 because there may be multiple ground truths
    if k == 1:
        recall = found_count
    else:
        recall = found_count / len(target_id_set)

    if verbose:
        print(f"\033[96mRecall@{k}: {found_count}/{len(target_id_set)} = {recall:.2f}\033[0m")
    
    return recall

def calculate_success_at_k(
    target_ids: list[str],
    retrieved_ids: list[str],
    k: int,
    verbose: bool = False
) -> int:
    """Calculate Success@k (Hit Rate@k).
    
    Args:
        target_ids: List of target document IDs (ground truth).
        retrieved_ids: List of retrieved document IDs.
        k: The number of top results to consider.
        
    Returns:
        int: 1 if at least one target_id is found in the top-k retrieved_ids,
             otherwise 0.
    """
    target_id_set = {str(id) for id in target_ids}
    retrieved_ids = [str(id) for id in retrieved_ids] if retrieved_ids else []

    retrieved_ids_at_k = retrieved_ids[:k]
    
    if verbose:
        print(f"\033[96mTarget IDs: {target_id_set}\033[0m")
        print(f"\033[92mRetrieved IDs @{k}: {retrieved_ids_at_k}\033[0m")

    # Success is binary: 1 if any overlap, 0 otherwise
    success = int(any(rid in target_id_set for rid in retrieved_ids_at_k))

    if verbose:
        print(f"\033[96mSuccess@{k}: {success}\033[0m")

    return success


def calculate_nDCG_at_k(
    target_ids: list[str], 
    retrieved_ids: list[str], 
    k: int, 
    verbose: bool = False
) -> float:
    """Calculate nDCG@k for retrieved documents with binary relevance.
    
    Args:
        target_ids: List of relevant document IDs
        retrieved_ids: List of retrieved document IDs in ranked order
        k: Number of top documents to consider
        verbose: Whether to print debug information
    
    Returns:
        nDCG@k score (0 to 1)
    """
    target_id_set = {str(id) for id in target_ids}
    retrieved_ids = [str(id) for id in retrieved_ids[:k]] if retrieved_ids else []
    
    # Calculate DCG@k - sum of (relevance / log2(position + 1))
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in target_id_set:
            # Position starts at 1, so we use i+2 for the denominator
            dcg += 1.0 / np.log2(i + 2) if i > 0 else 1.0
    
    # Calculate IDCG@k - best possible DCG if we had perfect ranking
    idcg = 0.0
    num_relevant = min(len(target_id_set), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2) if i > 0 else 1.0
    
    # Calculate nDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    if verbose:
        print(f"\033[96mTarget IDs: {target_id_set}\033[0m")
        print(f"\033[92mRetrieved IDs @{k}: {retrieved_ids}\033[0m")
        print(f"\033[93mDCG@{k}: {dcg:.4f}, IDCG@{k}: {idcg:.4f}\033[0m")
        print(f"\033[96mnDCG@{k}: {ndcg:.4f}\033[0m")
    
    return ndcg

def calculate_coverage(
    retrieved_ids: list[str], 
    nugget_data: list[dict], 
    k: int = 100, 
    verbose: bool = False
):
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
    
    retrieved_ids = [str(id) for id in retrieved_ids[:k]] if retrieved_ids else []
    
    covered_nuggets = set()
    nugget_coverage_details = []
    
    for i, nugget in enumerate(nugget_data):
        nugget_id = nugget.get('id', f'nugget_{i}')
        nugget_relevant_ids = [str(id) for id in nugget.get('relevant_corpus_ids', [])]
        
        covered = any(doc_id in retrieved_ids for doc_id in nugget_relevant_ids)
        
        if covered:
            covered_nuggets.add(nugget_id)
            nugget_coverage_details.append(f"\033[92mNugget {i+1}: Covered\033[0m")
        else:
            nugget_coverage_details.append(f"\033[91mNugget {i+1}: Not covered\033[0m")
    
    coverage_score = len(covered_nuggets) / len(nugget_data)
    
    if verbose:
        print(f"\033[96mCoverage@{k} evaluation:\033[0m")
        print(f"Total nuggets: {len(nugget_data)}")
        print(f"Covered nuggets: {len(covered_nuggets)}")

    if verbose:
        print(f"\033[96mCoverage@{k}: {len(covered_nuggets)}/{len(nugget_data)} = {coverage_score:.2f}\033[0m")
    
    return coverage_score


def calculate_alpha_ndcg(
    retrieved_ids: list[str], 
    nugget_data: list[dict], 
    alpha: float = 0.5, 
    k: int = 10, 
    verbose: bool = False
):
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
    
    retrieved_ids = [str(id) for id in retrieved_ids[:k]]
    
    covered_nuggets = set()
    dcg = 0.0
    position_gains = []
    
    for i, doc_id in enumerate(retrieved_ids):
        position = i + 1
        doc_gain = 0.0
        doc_nuggets = []
        
        for j, nugget in enumerate(nugget_data):
            nugget_id = nugget.get('id', f'nugget_{j}')
            nugget_relevant_ids = [str(id) for id in nugget.get('relevant_corpus_ids', [])]
            
            if doc_id in nugget_relevant_ids:
                if nugget_id not in covered_nuggets:
                    doc_gain += 1.0
                    covered_nuggets.add(nugget_id)
                    doc_nuggets.append(f"new:{nugget_id}")
                else:
                    doc_gain += (1 - alpha)
                    doc_nuggets.append(f"redundant:{nugget_id}")
        
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
    
    idcg = 0.0
    for i in range(min(len(nugget_data), k)):
        position = i + 1
        ideal_gain = 1.0
        idcg += ideal_gain / np.log2(position + 1)
    
    alpha_ndcg = dcg / idcg if idcg > 0 else 0.0
    
    if verbose:
        print(f"\033[96mα-nDCG@{k} evaluation (α={alpha}):\033[0m")
        print(f"Total nuggets: {len(nugget_data)}")
        print(f"Documents evaluated: {len(retrieved_ids)}") 
        print(f"\nDCG: {dcg:.3f}, IDCG: {idcg:.3f}")
        print(f"\033[96mα-nDCG@{k}: {alpha_ndcg:.3f}\033[0m")
    
    return alpha_ndcg