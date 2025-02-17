`uv` lookup with experimental weaviate agent clients:

```bash
uv add weaviate-client@git+https://github.com/cdpierse/weaviate-python-client.git@main
```

Look for this in `pyproject.toml`:

```
[dependencies]
weaviate-client = { git = "https://github.com/cdpierse/weaviate-python-client.git", rev = "main" }
```

...

```bash
uv install

uv run python3 benchmark_run.py
```
