import os
import yaml
import weaviate
import asyncio
import argparse
from pathlib import Path
from benchmarker.src.dataset import in_memory_dataset_loader
from benchmarker.src.database import database_loader
from benchmarker.src.agent import AgentBuilder
from benchmarker.src.query_agent_benchmark import (
    run_queries, 
    analyze_results,
    pretty_print_query_agent_benchmark_metrics,
    query_agent_benchmark_metrics_to_markdown
)
from benchmarker.src.utils import save_all_results, pretty_print_dict

def load_config(config_path: str):
    """Load main config and any agent-specific config if available."""
    # Load main config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load agent-specific config if it exists
    agent_config = {}
    agent_config_path = Path(os.path.dirname(__file__), "rag_ablation_config.yml")
    if config["agent_name"] == "rag-ablation":
        with open(agent_config_path) as f:
            agent_config = yaml.safe_load(f)
    
    return config, agent_config

async def main():
    parser = argparse.ArgumentParser(description='Run benchmark tests')
    parser.add_argument('--agents-host', type=str, default="https://api.agents.weaviate.io",
                        help='Host URL for agents API')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to test (overrides config value)')
    parser.add_argument('--experiment-name', type=str, default="query_agent_prod",
                        help='Name for this experiment run')
    args = parser.parse_args()
    
    config_path = Path(os.path.dirname(__file__), "config.yml")
    config, agent_config = load_config(config_path)
    
    # Extract agent-specific parameters with defaults
    agent_params = {
        "write_queries": agent_config.get("write_queries", False),
        "filter_results": agent_config.get("filter_results", False),
        "summarize_results": agent_config.get("summarize_results", False),
        "model_name": agent_config.get("model_name", "openai/gpt-4o")
    }

    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    documents, queries = in_memory_dataset_loader(config["dataset"])
    print("\033[92mFirst Document:\033[0m")
    pretty_print_dict(documents[0])
    print("\033[92mFirst Query\033[0m")
    pretty_print_dict(queries[0])

    if config["reload_database"]:
        database_loader(weaviate_client, config["dataset"], documents)

    query_agent = AgentBuilder(
        dataset_name=config["dataset"],
        agent_name=config["agent_name"],
        agents_host=args.agents_host,
        **agent_params
    )

    num_samples = args.num_samples if args.num_samples is not None else 5
    
    results = run_queries(
        queries=queries,
        agent_name=config["agent_name"],
        query_agent=query_agent,
        num_samples=num_samples
    )

    save_all_results(
        results=results, 
        config=config,
        experiment_name=args.experiment_name,
        agents_host=args.agents_host,
        num_samples=num_samples
    )

    metrics = await analyze_results(weaviate_client, config["dataset"], results, queries)

    pretty_print_query_agent_benchmark_metrics(metrics)
    
    query_agent_benchmark_metrics_to_markdown(
        metrics=metrics,
        dataset_name=config["dataset"],
        experiment_name=args.experiment_name
    )

    weaviate_client.close()

if __name__ == "__main__":
    asyncio.run(main())