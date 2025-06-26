import os
import yaml
import weaviate
import asyncio
import argparse
from pathlib import Path
from benchmarker.src.agent import AgentBuilder
from benchmarker.src.dataset import in_memory_dataset_loader

from benchmarker.src.query_agent_benchmark import (
    run_queries,
    run_queries_async,
    analyze_results
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
    args = parser.parse_args()
    
    use_async = True

    config_path = Path(os.path.dirname(__file__), "config.yml")
    config = load_config(config_path)

    _, queries = in_memory_dataset_loader(config["dataset"])
    print("\033[92mFirst Query\033[0m")
    pretty_print_dict(queries[0])

    query_agent = AgentBuilder(
        dataset_name=config["dataset"],
        agent_name=config["agent_name"],
        agents_host=args.agents_host,
        use_async=use_async,
    )

    num_samples = config["num_samples"]
    
    # Run queries based on async flag
    if use_async:        
        print("\033[92mRunning queries async!\033[0m")
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

    # Save all results to JSON file
    # save_all_results(
    #     results=results, 
    #     config=config,
    #     agent_name=config["experiment_name"],
    #     agents_host=args.agents_host,
    #     num_samples=num_samples
    # )

    metrics = await analyze_results(
        weaviate_client, 
        config["dataset"], 
        results, 
        queries,
        run_lm_judge=config["run_lm_judge"]
    )

    print(metrics)

    weaviate_client.close()

if __name__ == "__main__":
    asyncio.run(main())