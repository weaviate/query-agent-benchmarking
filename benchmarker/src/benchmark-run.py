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
    run_queries_async,
    analyze_results,
    pretty_print_query_agent_benchmark_metrics,
    query_agent_benchmark_metrics_to_markdown
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
    parser.add_argument('--use-async', type=bool, default=True,
                        help='Use async query processing for better performance')
    parser.add_argument('--experiment-name', type=str, default="query_agent_prod",
                        help='Name for this experiment run')
    args = parser.parse_args()
    
    config_path = Path(os.path.dirname(__file__), "config.yml")
    config = load_config(config_path)

    # Initialize Weaviate client (sync version for data loading)
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

    # Close the sync client as we'll use async for queries
    weaviate_client.close()

    # Create agent builder with async support if requested
    query_agent = AgentBuilder(
        dataset_name=config["dataset"],
        agent_name=config["agent_name"],
        agents_host=args.agents_host,
        use_async=args.use_async,
    )

    num_samples = args.num_samples if args.num_samples is not None else 5
    
    # Run queries based on async flag
    if args.use_async:        
        # Initialize async agent
        await query_agent.initialize_async()
        
        try:
            results = await run_queries_async(
                queries=queries,
                query_agent=query_agent,
                num_samples=num_samples,
                batch_size=config["batch_size"],
                max_concurrent=config["max_concurrent"]
            )
        finally:
            # Clean up async connection
            await query_agent.close_async()
    else:
        print("\n\033[94mRunning synchronous benchmark\033[0m")
        results = run_queries(
            queries=queries,
            query_agent=query_agent,
            num_samples=num_samples
        )

    # Re-open sync client for analysis
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    # save_all_results(
    #     results=results, 
    #     config=config,
    #     agent_name=config["agent_name"],
    #     agents_host=args.agents_host,
    #     num_samples=num_samples
    # )

    metrics = await analyze_results(weaviate_client, config["dataset"], results, queries)

    pretty_print_query_agent_benchmark_metrics(
        metrics, 
        dataset_name=config["dataset"],
        experiment_name=args.experiment_name + (" (async)" if args.use_async else " (sync)")
    )
    
    query_agent_benchmark_metrics_to_markdown(
        metrics=metrics,
        dataset_name=config["dataset"],
        agent_name=config["agent_name"] + (" (async)" if args.use_async else " (sync)")
    )

    weaviate_client.close()

if __name__ == "__main__":
    asyncio.run(main())