import weaviate
from weaviate.agents.query import QueryAgent
import os

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
)

print("Client ready:", weaviate_client.is_ready())

qa = QueryAgent(
    client=weaviate_client,
    collections=["FreshstackLangchain"],
    agents_host="https://dev-agents.labs.weaviate.io"
)

searcher = qa.prepare_search(
    query="How do I use LangChain with Weaviate?",
)
print("Searcher prepared successfully!")
print(f"Agent URL: {searcher.agent_url}")
print(f"Search endpoint: {searcher.agent_url}/search_only")

search_response = searcher.execute(
    limit=10,
    offset=0
)
    
print("Search completed successfully!")
print(f"Response type: {type(search_response)}")
    
if hasattr(search_response, '__dict__'):
    print(f"Response attributes: {list(search_response.__dict__.keys())}")

objs = []
for obj in search_response.search_results.objects:
    objs.append(obj.properties["dataset_id"])

print(objs)