import os
from pathlib import Path
import yaml

import weaviate

from benchmarker.src.dataset import in_memory_dataset_loader
from benchmarker.src.database import database_loader
from benchmarker.src.utils import pretty_print_dict

def load_config(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
)

config_path = Path(os.path.dirname(__file__), "config.yml")
config = load_config(config_path)

documents, _ = in_memory_dataset_loader(config["dataset"])
print("\033[92mFirst Document:\033[0m")
pretty_print_dict(documents[0])

# NOTE [Named Vectors]: Update this to pass in the vectorizers that will be used, e.g.
# `database_loader(..., vectorizers=[arctic-2.0, arctic-1.5, embedv4, colbert-gte, ...])`
database_loader(weaviate_client, config["dataset"], documents)

weaviate_client.close()