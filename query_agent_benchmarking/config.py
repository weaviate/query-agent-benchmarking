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
    "lotte/lifestyle/test/forum",
    "lotte/lifestyle/test/search",
    "lotte/recreation/test/forum",
    "lotte/recreation/test/search",
    "wixqa"
)

def print_supported_datasets():
    print("Supported datasets:")
    for dataset in supported_datasets:
        print(f"- {dataset}")