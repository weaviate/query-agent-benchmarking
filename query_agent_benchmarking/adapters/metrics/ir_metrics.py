"""IR metrics adapter - re-exports from the original metrics module.

The actual metric functions remain in query_agent_benchmarking.metrics.ir_metrics.
This adapter provides a consistent import path within the hexagonal architecture.
"""

from query_agent_benchmarking.metrics.ir_metrics import (
    calculate_recall_at_k,
    calculate_success_at_k,
    calculate_nDCG_at_k,
    calculate_coverage,
    calculate_alpha_ndcg,
)

__all__ = [
    "calculate_recall_at_k",
    "calculate_success_at_k",
    "calculate_nDCG_at_k",
    "calculate_coverage",
    "calculate_alpha_ndcg",
]
