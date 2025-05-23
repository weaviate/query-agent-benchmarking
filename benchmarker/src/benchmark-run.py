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
    pretty_print_query_agent_benchmark_metrics
)
from benchmarker.src.utils import pretty_print_dict

def load_config(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

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
    config = load_config(config_path)

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
    )

    num_samples = args.num_samples if args.num_samples is not None else 5
    
    results = run_queries(
        queries=queries,
        query_agent=query_agent,
        num_samples=num_samples
    )

    # save_all_results(
    #     results=results, 
    #     config=config,
    #     agent_name=config["agent_name"],
    #     agents_host=args.agents_host,
    #     num_samples=num_samples
    # )

    metrics = await analyze_results(weaviate_client, config["dataset"], results, queries)

    pretty_print_query_agent_benchmark_metrics(metrics)
    
    # query_agent_benchmark_metrics_to_markdown(
    #     metrics=metrics,
    #     dataset_name=config["dataset"],
    #     agent_name=config["agent_name"]
    # )

    weaviate_client.close()

if __name__ == "__main__":
    asyncio.run(main())