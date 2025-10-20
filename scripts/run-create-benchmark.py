import query_agent_benchmarking

# Create the benchmark
result = query_agent_benchmarking.create_benchmark(
    docs_source_collection="BrightBiology",           # Your source collection
    benchmark_collection_name="SyntheticBrightBiology",  # New benchmark collection
    query_property_name="simulated_user_query",       # Name for generated queries
    content_property_name="content",                  # Property with document text
    id_property_name="dataset_id",                    # Property with document ID
    num_queries=100,
    delete_if_exists=True,                            # Clean slate each run
)