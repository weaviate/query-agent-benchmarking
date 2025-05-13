import json
import os
import time
import yaml
import weaviate
import asyncio
from pathlib import Path
from benchmarker.cmd.dataset import in_memory_dataset_loader
from benchmarker.cmd.database import database_loader
from benchmarker.cmd.agent import QueryAgentBuilder
from benchmarker.cmd.query_agent_benchmark import run_queries, analyze_results

async def main():
    config_path = Path(os.path.dirname(__file__), "config.yml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    config = yaml.safe_load(open(config_path))

    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    documents, queries = in_memory_dataset_loader(config["dataset"])
    print("\033[92mFirst Document:\033[0m")
    print(documents[0])
    print("\033[92mFirst Query\033[0m")
    print(queries[0])

    if config["reload_database"]:
        database_loader(weaviate_client, config["dataset"], documents)

    query_agent = QueryAgentBuilder(
        weaviate_client,
        config["dataset"]
    )

    results = run_queries(
        queries,
        query_agent,
        config["test_samples"]
    )

    await analyze_results(weaviate_client, config["dataset"], results, queries)

if __name__ == "__main__":
    asyncio.run(main())