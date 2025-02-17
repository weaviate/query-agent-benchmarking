import weaviate
import yaml
import time
import json

from cmd.dataset import load_dataset
from cmd.database import load_database
from cmd.run_agent import AgentWrapper

config = yaml.safe_load(open("config.yml"))

# 1. load database
load_database(config.dataset)

# 2. load dataset
dataset = load_dataset(config.dataset)

# 3. agent inference wrapper -- read config options
agent = AgentWrapper(config=config.agent_name)

# 4. loop through dataset and populate the experiment data
results = []
start = time.time()
for row in dataset:
    # will need to update the `collections` argument to `QueryAgent`
    result = agent.run(**row)
    results.append(result)

experiment_time = time.time() - start
print(f"Experiment completed in {experiment_time} seconds.")

# 4. save the experiment data to disk
with open("results.json", "w") as w:
    json.dump(results, w)