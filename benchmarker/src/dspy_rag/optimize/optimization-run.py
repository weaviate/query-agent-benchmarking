import time

import dspy

from benchmarker.src.dataset import in_memory_dataset_loader
from benchmarker.src.dspy_rag.rag_programs import RAG_VARIANTS
from benchmarker.src.dspy_rag.metrics import create_metric
from benchmarker.src.dspy_rag.data_preparation import create_dspy_examples_from_dataset
from benchmarker.src.dspy_rag.utils import (
    load_optimization_config,
    setup_dspy,
    setup_weaviate,
    print_configuration_summary
)
from benchmarker.src.dspy_rag.data_preparation import (
    get_collection_info
)
from benchmarker.src.dspy_rag.dspy_optimizer import create_optimizer

import logging
logging.disable(logging.CRITICAL)

# mlflow ui --port 5000
#mlflow.set_experiment("DSPy")
#mlflow.dspy.autolog()

opt_config = load_optimization_config("./optimization_config.yml")
print_configuration_summary(opt_config)
setup_dspy()
weaviate_client = setup_weaviate()

_, queries = in_memory_dataset_loader(opt_config["dataset_name"])

train_examples, test_examples = create_dspy_examples_from_dataset(
    queries = queries,
    max_train = opt_config["max_train"],
    max_test = opt_config["max_test"]
) 

collection_name, target_property_name = get_collection_info(
    opt_config["dataset_name"]
)
        
rag_class = RAG_VARIANTS[opt_config["agent_name"]]
rag_program = rag_class(
    collection_name=collection_name,
    target_property_name=target_property_name
)

metric = create_metric(
    metric_type=opt_config["metric_type"],
    weaviate_client=weaviate_client,
    dataset_name=opt_config["dataset_name"],
)

eval_kwargs = dict(
    num_threads=1,
    display_progress=True, 
    display_table=0
)

evaluator = dspy.Evaluate(
    devset=test_examples, 
    metric=metric, 
    num_threads=1,
    display_progress=True,
    max_errors=1,
    provide_traceback=True
)

score = evaluator(rag_program, **eval_kwargs)
print("\033[92m" + "="*50 + "\033[0m")
print(f"Uncompiled score: {score}")
print("\033[92m" + "="*50 + "\033[0m")

optimizer = create_optimizer(opt_config, metric)

optimization_start = time.time()

# ToDo -- wrap this in the returned optimizer from `create_optimizer`
if opt_config["optimizer_type"] == "copro":
    compiled_program = optimizer.compile(
        student=rag_program,
        trainset=train_examples,
        eval_kwargs=eval_kwargs
    )
else:
    compiled_program = optimizer.compile(
        student=rag_program,
        trainset=train_examples
    )

compiled_program.save("optimized.json")

print(f"Optimization ran in {time.time() - optimization_start}")

score = evaluator(compiled_program, **eval_kwargs)
print("\033[92m" + "="*50 + "\033[0m")
print(f"Compiled score: {score}")
print("\033[92m" + "="*50 + "\033[0m")