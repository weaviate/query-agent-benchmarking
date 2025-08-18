import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import random
import yaml

import weaviate

from benchmarker.agent import AgentBuilder
from benchmarker.dataset import in_memory_dataset_loader
from benchmarker.query_agent_benchmark import (
    run_queries,
    run_queries_async,
    analyze_results
)
from benchmarker.utils import pretty_print_dict

def load_config(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

async def main():
    config_path = Path(os.path.dirname(__file__), "config.yml")
    config = load_config(config_path)
    
    agents_host = config.get("agents_host", "https://api.agents.weaviate.io")
    use_async = config.get("use_async", True)

    _, queries = in_memory_dataset_loader(config["dataset"])
    print(f"There are \033[92m{len(queries)}\033[0m total queries in this dataset.\n")
    print("\033[92mFirst Query\033[0m")
    pretty_print_dict(queries[0])

    random.seed(42)
    random.shuffle(queries)
    print("Queries have been shuffled for fair comparison (seed=42).\n")

    query_agent = AgentBuilder(
        dataset_name=config["dataset"],
        agent_name=config["agent_name"],
        agents_host=agents_host,
        use_async=use_async,
    )

    num_samples = config["num_samples"]
    
    if use_async:        
        print("\033[92mRunning queries async!\033[0m")
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
            await query_agent.close_async()
    else:
        print("\n\033[94mRunning synchronous benchmark\033[0m")
        results = run_queries(
            queries=queries,
            query_agent=query_agent,
            num_samples=num_samples
        )

    # Open sync client for analysis
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    metrics = await analyze_results(
        weaviate_client, 
        config["dataset"], 
        results,
        queries,
    )

    metrics["timestamp"] = datetime.now().isoformat()
    dataset_name = config["dataset"].replace("/", "-")
    with open(f"{dataset_name}-{config['agent_name']}-{config['num_samples']}-results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)

    weaviate_client.close()

if __name__ == "__main__":
    asyncio.run(main())