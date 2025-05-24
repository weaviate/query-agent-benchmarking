### Start FastAPI server

#### Query Agent

```
python3 query_agent_server.py
```

#### RAG Ablations

```
python3 rag_server.py
```

### Run Locust Test

```
locust -f run_locust_test.py --users 10 --spawn-rate 2 --host http://localhost:8001
```