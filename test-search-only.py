import weaviate
from weaviate.auth import Auth
from weaviate.agents.query import QueryAgent
import os

# Connect to Weaviate Cloud
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
)

print("Client ready:", weaviate_client.is_ready())

# Create the QueryAgent
qa = QueryAgent(
    client=weaviate_client,
    collections=["FreshstackLangchain"],
    agents_host="https://dev-agents.labs.weaviate.io"
)

searcher = qa.prepare_search(
    query="How do I use LangChain with Weaviate?",
    # filters=None,  # Optional: you can add filters here
    # collections=["FreshstackLangchain"]  # Optional: override collections
)
print("Searcher prepared successfully!")
print(f"Agent URL: {searcher.agent_url}")
print(f"Search endpoint: {searcher.agent_url}/search_only")

search_response = searcher.execute(
    limit=10,    # Number of results to return
    offset=0     # Starting position (for pagination)
)
    
print("Search completed successfully!")
print(f"Response type: {type(search_response)}")
    
if hasattr(search_response, '__dict__'):
    print(f"Response attributes: {list(search_response.__dict__.keys())}")

print(search_response.search_results)

