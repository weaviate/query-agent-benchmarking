def qa_source_parser(
    query_agent_sources_response,
    collection
):
    if not query_agent_sources_response:
        return []
    
    sources = query_agent_sources_response
    source_uuids = [source.object_id for source in sources]
    
    matching_objects = collection.query.fetch_objects_by_ids(
        source_uuids
    )
    
    dataset_ids = []
    for o in matching_objects.objects:
        dataset_id = o.properties.get('dataset_id')
        if dataset_id is not None:
            # Ensure dataset_id is added as a string to the list
            dataset_ids.append(str(dataset_id))
    
    return dataset_ids

def get_object_by_dataset_id(dataset_id, objects_list):
    """Retrieve an object by its dataset_id from the objects list."""
    for obj in objects_list:
        if obj["dataset_id"] == dataset_id:
            return obj
    return None

def make_json_serializable(obj):
    """Convert objects to JSON serializable formats."""
    import inspect
    
    if hasattr(obj, '__dict__'):
        # Handle class instances by converting to dict
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_') and not inspect.ismethod(v)}
    elif isinstance(obj, dict):
        # Recursively convert dict values
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert list items
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # These types are already JSON serializable
        return obj
    else:
        # Try to convert to string for other types
        try:
            return str(obj)
        except Exception:
            return f"<Non-serializable object of type {type(obj).__name__}>"

def save_all_results(
    results: dict, 
    config: dict, 
    experiment_name: str = "query_agent_prod", 
    agents_host: str = None,
    num_samples: int = None
):
    """Save benchmark results to a file.
    
    Args:
        results: Dictionary containing benchmark results
        config: Dictionary containing benchmark configuration
        experiment_name: Name of this experiment run
        agents_host: Host URL for agents API
        num_samples: Number of samples tested
    """
    import json
    import os
    from datetime import datetime
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Convert results to JSON serializable format
    serializable_results = make_json_serializable(results)
    
    # Combine results with relevant config information
    output = {
        "results": serializable_results,
        "config": {
            "dataset": config.get("dataset"),
            "agent_name": config.get("agent_name"),
            "experiment_name": experiment_name,
            "agents_host": agents_host,
            "num_samples": num_samples,
            "timestamp": timestamp
        }
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {filepath}")
    return filepath
    
def pretty_print_dict(dict_to_print: dict):
    """
    Pretty prints a dictionary with colored keys and formatted output.
    
    Args:
        dict_to_print: Dictionary to be printed
    """
    print("="*60)
    for key, value in dict_to_print.items():
        # Print key in cyan, value in white
        print(f"\t\033[96m{key}\033[0m: {value}")
    print("="*60)
    print("\n\n")