from dotenv import load_dotenv

from .experimental.add_hard_negatives import add_hard_negatives

# Search benchmark exports
from .internal.core.services import run_search_eval, run_search_evals, compare_search_agents

# Ask benchmark exports
from .internal.core.services import run_ask_eval

from .internal.core.services import compare_embeddings
from .internal.core.services import populate_db, populate_db_multi
from .internal.adapters.database import database_loader
from .internal.adapters.dataset import (
    in_memory_dataset_loader,
    in_memory_ask_dataset_loader,
    load_ask_queries_from_weaviate,
    load_longmemeval_docs_by_tenant,
)

# Models
from .internal.core.domain.models import (
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
from .internal.core.ports.search_agent import SearchAgent
from .internal.adapters.agents.factory import create_search_agent, create_ask_agent
from .internal.adapters.agents.engram_dspy_agent import EngramDSPyAgent

# Metrics
from .internal.adapters.metrics.ir_metrics import (
    calculate_recall_at_k,
    calculate_success_at_k,
    calculate_nDCG_at_k,
    calculate_coverage,
    calculate_alpha_ndcg,
)
from .internal.adapters.metrics.lmjudge_alignment import LMJudge
from .internal.adapters.metrics.exact_match import calculate_exact_match
from .internal.adapters.metrics.officeqa_metric import (
    score_answer as officeqa_score_answer,
)

from .experimental.create_benchmark import create_benchmark
from .internal.config.config import (
    print_supported_datasets,
    print_supported_ask_datasets,
    print_named_vector_targets,
    get_named_vector_target_entry,
    resolve_named_vector_target,
    resolve_embedding_providers,
    supported_search_datasets,
    supported_ask_datasets,
)
from .internal.adapters.results.serialization import (
    save_trial_results,
    save_ask_trial_results,
    save_trial_metrics,
    save_aggregated_results,
)

load_dotenv()

__all__ = [
    # Main entry points
    "run_search_eval",
    "run_search_evals",
    "compare_search_agents",
    "run_ask_eval",
    # Utilities
    "add_hard_negatives",
    "populate_db",
    "populate_db_multi",
    "database_loader",
    "in_memory_dataset_loader",
    "in_memory_ask_dataset_loader",
    "load_ask_queries_from_weaviate",
    "load_longmemeval_docs_by_tenant",
    "compare_embeddings",
    "create_benchmark",
    "print_supported_datasets",
    "print_supported_ask_datasets",
    "print_named_vector_targets",
    "get_named_vector_target_entry",
    "resolve_named_vector_target",
    "resolve_embedding_providers",
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
    "SearchAgent",
    "create_search_agent",
    "create_ask_agent",
    "EngramDSPyAgent",
    # Metrics - IR
    "calculate_recall_at_k",
    "calculate_success_at_k",
    "calculate_nDCG_at_k",
    "calculate_coverage",
    "calculate_alpha_ndcg",
    # Metrics - LLM Judge
    "LMJudge",
    # Metrics - Exact Match
    "calculate_exact_match",
    # Metrics - OfficeQA
    "officeqa_score_answer",
    # Result serialization
    "save_trial_results",
    "save_ask_trial_results",
    "save_trial_metrics",
    "save_aggregated_results",
]
__version__ = "0.5"
