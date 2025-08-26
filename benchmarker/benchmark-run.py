import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import yaml

import weaviate

from benchmarker.agent import AgentBuilder
from benchmarker.dataset import in_memory_dataset_loader
from benchmarker.query_agent_benchmark import (
    run_queries,
    run_queries_async,
    analyze_results,
    aggregate_metrics
)
from benchmarker.utils import pretty_print_dict
from benchmarker.config import supported_datasets

def load_config(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

async def main():
    config_path = Path(os.path.dirname(__file__), "benchmark-config.yml")
    config = load_config(config_path)
    
    agents_host = config.get("agents_host", "https://api.agents.weaviate.io")
    use_async = config.get("use_async", True)

    if config["dataset"] not in supported_datasets:
        raise ValueError(f"Dataset {config['dataset']} is not supported. Supported datasets are: {supported_datasets}")

    _, queries = in_memory_dataset_loader(config["dataset"])
    print(f"There are \033[92m{len(queries)}\033[0m total queries in this dataset.\n")
    print("\033[92mFirst Query\033[0m")
    pretty_print_dict(queries[0])

    if config["use_subset"]:
        import random
        random.seed(42)
        random.shuffle(queries)
        queries = queries[:config["num_samples"]]
        print(f"Using a subset of {config['num_samples']} queries.")

    query_agent = AgentBuilder(
        dataset_name=config["dataset"],
        agent_name=config["agent_name"],
        agents_host=agents_host,
        use_async=use_async,
    )

    num_trials = config["num_trials"]

    metrics_across_trials = []

    # NOTE: We are using multiple trials to account for the stochasticity of LLM-based systems.
    for trial in range(num_trials):
        print(f"\033[92mRunning trial {trial+1}/{num_trials}\033[0m")

        if use_async:        
            print("\033[92mRunning queries async!\033[0m")
            await query_agent.initialize_async()
            
            try:
                results = await run_queries_async(
                    queries=queries,
                    query_agent=query_agent,
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
            )

        # Open sync client for analysis
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )

        metrics = await analyze_results(
            config["dataset"], 
            results,
            queries,
        )
        print(metrics)

        metrics_across_trials.append(metrics)

    weaviate_client.close()

    aggregated_metrics = aggregate_metrics(metrics_across_trials)

    metrics["timestamp"] = datetime.now().isoformat()
    dataset_name = config["dataset"].replace("/", "-")
    with open(f"{dataset_name}-{config['agent_name']}-{config['num_trials']}-results.json", "w") as f:
        json.dump(aggregated_metrics, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())