def qa_source_parser(
    query_agent_sources_response,
    collection
):
    sources = query_agent_sources_response,
    source_uuids = [source.object_id for source in sources]
    
    matching_objects = collection.query.fetch_objects_by_ids(
        source_uuids
    )
    
    dataset_ids = []
    for o in matching_objects.objects:
        dataset_ids.append(o.properties.get('dataset_id'))
    
    return dataset_ids