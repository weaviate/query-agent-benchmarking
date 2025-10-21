from query_agent_benchmarking.models import InMemoryQuery

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
    elif isinstance(obj, InMemoryQuery):
        return {
            "question": obj.question,
            "dataset_ids": obj.dataset_ids
        }
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
    
def pretty_print_in_memory_query(in_memory_query: InMemoryQuery):
    """
    Pretty prints a dictionary with colored keys and formatted output.
    
    Args:
        dict_to_print: Dictionary to be printed
    """
    print(f"\t\033[96mQuestion\033[0m: {in_memory_query.question}")
    print(f"\t\033[96mDataset IDs\033[0m: {in_memory_query.dataset_ids}")
    print("="*60)
    print("\n\n")