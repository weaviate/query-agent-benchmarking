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
