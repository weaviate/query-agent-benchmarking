supported_search_datasets = (
    "beir/fiqa/test",
    "beir/nq",
    "beir/scifact/test",
    "bright/biology",
    "bright/earth_science",
    "bright/economics",
    "bright/psychology",
    "bright/robotics",
    "enron",
    "irpapers/images",
    "irpapers/text",
    "irpapers/images/visual-queries",
    "irpapers/text/visual-queries",
    "lotte/lifestyle/test/forum",
    "lotte/lifestyle/test/search",
    "lotte/recreation/test/forum",
    "lotte/recreation/test/search",
    "wixqa"
)

supported_ask_datasets = (
    "irpapers/text",    # Uses IRPapersText_Default collection
    "irpapers/images",  # Uses IRPapersImages_Default collection
)

supported_embedding_models = (
    "cohere/embed-v4.0",
    "voyageai/voyage-3-large",
    "weaviate/Snowflake/snowflake-arctic-embed-m-v1.5",
    "weaviate/Snowflake/snowflake-arctic-embed-l-v2.0",
)

def print_supported_datasets():
    """Print all supported search datasets."""
    print("Supported search datasets:")
    for dataset in supported_search_datasets:
        print(f"  - {dataset}")

def print_supported_ask_datasets():
    """Print all supported ask datasets."""
    print("Supported ask datasets:")
    for dataset in supported_ask_datasets:
        print(f"  - {dataset}")