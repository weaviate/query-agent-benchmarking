from dspy.teleprompt import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    COPRO,
    MIPROv2
)

def create_optimizer(config, metric):
    """Create the DSPy optimizer."""
    print(f"\033[94mCreating {config['optimizer_type']} optimizer\033[0m")
    
    if config["optimizer_type"] == "bootstrap_few_shot":
        return BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=config["max_bootstrapped_demos"],
            max_labeled_demos=config["max_labeled_demos"]
        )
    
    elif config["optimizer_type"] == "bootstrap_random_search":
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=config["max_bootstrapped_demos"],
            max_labeled_demos=config["max_labeled_demos"],
            num_candidate_programs=config["num_candidate_programs"]
        )
    
    elif config["optimizer_type"] == "copro":
        return COPRO(
            metric=metric,
            breadth=config["breadth"],
            depth=config["depth"],
            init_temperature=config["init_temperature"],
            verbose=True
        )
    
    elif config["optimizer_type"] == "mipro":
        return MIPROv2(
            metric=metric,
            verbose=True
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {config['optimizer_type']}")