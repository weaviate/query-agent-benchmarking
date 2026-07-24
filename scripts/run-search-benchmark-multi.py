"""Run the search benchmark across multiple datasets.

Mirrors scripts/populate-db-multi.py: the dataset list comes from
``search_datasets`` in benchmark-config.yml (the same "list in config" idea as
``dataset_names`` for populate_db_multi). Each dataset is evaluated and a
cross-dataset summary is printed.

If ``effort_sweep: true`` is set in the config, every dataset is additionally
swept across the "low"/"medium"/"high" effort levels and a per-dataset effort
comparison is printed.

    uv run python3 scripts/run-search-benchmark-multi.py
"""

import query_agent_benchmarking

query_agent_benchmarking.run_search_evals()
