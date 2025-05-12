import json
import os
import time
import yaml
import weaviate
from benchmarker.cmd.dataset import load_dataset
from benchmarker.cmd.database import load_database
from benchmarker.cmd.agent import QueryAgentBuilder
from benchmarker.cmd.query_agent_benchmark import run_queries, analyze_results

config = yaml.safe_load(open("config.yml"))

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
)

load_database(config.dataset)

queries = load_dataset(config.dataset)

query_agent = QueryAgentBuilder(
    weaviate_client,
    config.dataset
)

results = run_queries(
    queries,
    query_agent,
    config.test_samples
)

analyze_results(weaviate_client, results, queries)