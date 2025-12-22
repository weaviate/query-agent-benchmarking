from .experimental.add_hard_negatives import add_hard_negatives

# Search benchmark exports
from .search_benchmark_run import run_search_eval, run_search_evals

# Ask benchmark exports
from .ask_benchmark_run import run_ask_eval

from .compare_embeddings import compare_embeddings
from .database import database_loader
from .dataset import (
    in_memory_dataset_loader,
    in_memory_ask_dataset_loader,
    load_ask_queries_from_weaviate,
)

# Models
from .models import (
    DocsCollection,
    QueriesCollection,
    InMemoryQuery,
    ObjectID,
    QueryResult,
    # Search-specific
    InMemorySearchQuery,
    SearchResult,
    # Ask-specific
    InMemoryAskQuery,
    AskResult,
    AskQueriesCollection,
)

# Agent exports
from .agent import (
    SearchAgentBuilder,
    AskAgentBuilder,
    BaseAgentBuilder,
)

# Metrics
from .metrics import (
    # IR Metrics
    calculate_recall_at_k,
    calculate_success_at_k,
    calculate_nDCG_at_k,
    calculate_coverage,
    calculate_alpha_ndcg,
    # LLM Judge
    LMJudge,
    calculate_alignment_score,
)

from .experimental.create_benchmark import create_benchmark
from .config import (
    print_supported_datasets,
    print_supported_ask_datasets,
    supported_search_datasets,
    supported_ask_datasets,
)
from .result_serialization import save_trial_results, save_trial_metrics, save_aggregated_results

__all__ = [
    # Main entry points
    "run_search_eval",
    "run_search_evals",
    "run_ask_eval",
    # Utilities
    "add_hard_negatives",
    "database_loader",
    "in_memory_dataset_loader",
    "in_memory_ask_dataset_loader",
    "load_ask_queries_from_weaviate",
    "compare_embeddings",
    "create_benchmark",
    "print_supported_datasets",
    "print_supported_ask_datasets",
    "supported_search_datasets",
    "supported_ask_datasets",
    # Models
    "DocsCollection",
    "QueriesCollection",
    "InMemoryQuery",
    "ObjectID",
    "QueryResult",
    "InMemorySearchQuery",
    "SearchResult",
    "InMemoryAskQuery",
    "AskResult",
    "AskQueriesCollection",
    # Agents
    "SearchAgentBuilder",
    "AskAgentBuilder",
    "BaseAgentBuilder",
    # Metrics - IR
    "calculate_recall_at_k",
    "calculate_success_at_k",
    "calculate_nDCG_at_k",
    "calculate_coverage",
    "calculate_alpha_ndcg",
    # Metrics - LLM Judge
    "LMJudge",
    "calculate_alignment_score",
    # Result serialization
    "save_trial_results",
    "save_trial_metrics",
    "save_aggregated_results",
]
__version__ = "0.5"