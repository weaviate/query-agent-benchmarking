supported_datasets = (
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

supported_embedding_models = (
    "cohere/embed-v4.0",
    "voyageai/voyage-3-large",
    "weaviate/Snowflake/snowflake-arctic-embed-m-v1.5",
    "weaviate/Snowflake/snowflake-arctic-embed-l-v2.0",
)

def print_supported_datasets():
    print("Supported datasets:")
    for dataset in supported_datasets:
        print(f"- {dataset}")