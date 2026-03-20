from engram import EngramClient
import os
import uuid

client = EngramClient(
    api_key=os.getenv("ENGRAM_API_KEY"),
    base_url="https://dev-engram.labs.weaviate.io"
)

user_id = f"connor-1234567890"

'''
run = client.memories.add(
    "My favorite mattress brand is Purple",
    user_id=user_id,
)

print(f"Run ID: {run.run_id}")
print(f"Status: {run.status}")
'''

results = client.memories.search(
    query="What is the user's favorite mattress brand?",
    user_id=user_id,
)

for result in results:
    print(result)

"""
# Returns:
Memory(id='08505722-6e44-43b5-bb17-2091a58fa834', project_id='019ce79c-d6d6-7dea-8687-edaacade55ee', content="The user's favorite mattress brand is Purple.", topic='UserKnowledge', group='default', created_at='2026-03-19T13:15:46.668Z', updated_at='2026-03-19T13:15:46.668Z', user_id='connor-1234567890', conversation_id=None, tags=None, score=1)
"""