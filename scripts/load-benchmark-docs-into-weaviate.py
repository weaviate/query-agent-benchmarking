import os
from pathlib import Path

import weaviate

from query_agent_benchmarking.dataset import in_memory_dataset_loader
from query_agent_benchmarking.database import database_loader
from query_agent_benchmarking.utils import pretty_print_in_memory_query, load_config

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
)

config_path = Path(os.path.dirname(__file__), "benchmark-config.yml")
config = load_config(config_path)

documents, _ = in_memory_dataset_loader(config["dataset"])
print("\033[92mFirst Document:\033[0m")
pretty_print_in_memory_query(documents[0])

database_loader(weaviate_client, config["dataset"], documents)

weaviate_client.close()

