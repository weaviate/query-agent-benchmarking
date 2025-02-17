from cmd.dataset import load_dataset

dataset = load_dataset("./weaviate-gorilla.json")

print(len(dataset))