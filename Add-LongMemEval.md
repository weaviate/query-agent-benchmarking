# Add LongMemEval

Add the LongMemEval to `query-agent-benchmarking`.

- Extend the `database` layer to create a Multi-Tenant Weaviate Collection.

Here's how to create a multi-tenant collection using the Weaviate Python v4 client:

from weaviate.classes.config import Configure

multi_collection = client.collections.create(
    name="MultiTenancyCollection",
    # Enable multi-tenancy on the new collection
    multi_tenancy_config=Configure.multi_tenancy(enabled=True)
)
[Enable multi-tenancy]

Once the collection is created, you can add tenants to it:

from weaviate.classes.tenants import Tenant

# Add two tenants to the collection
multi_collection.tenants.create(
    tenants=[
        Tenant(name="tenantA"),
        Tenant(name="tenantB"),
    ]
)
[Add tenants manually]

You can also enable additional options like auto_tenant_creation (available from v1.25.0), which automatically creates a new tenant if you try to insert an object into a non-existent one:

from weaviate.classes.config import Configure

multi_collection = client.collections.create(
    name="CollectionWithAutoMTEnabled",
    multi_tenancy_config=Configure.multi_tenancy(
        enabled=True,
        auto_tenant_creation=True
    )
)
[Auto tenant creation]

Note: Tenant names can only contain alphanumeric characters (a-z, A-Z, 0-9), underscores (_), and hyphens (-), with a length of 4 to 64 characters.

The LongMemEval multi-tenant collection should have this schema:
- haystack_session
- haystack_session_id
- haystack_session_date

I created this script to help you understand the data further:

Connors-MacBook-Pro-2:eda cshorten$ uv run python3 longmemeval-s-eda.py
question_id: str
question_type: str
question: str
question_date: str
answer: str
answer_session_ids: list
haystack_dates: list
haystack_session_ids: list
haystack_sessions: list

...

For each of the 500 questions, create a tenant, ingest that question's haystack, run the question as a retrieval query against that tenant, compare results to answer_session_ids, then move to the next question. It's a straightforward loop but worth spelling out.

The evaluation labels are at the session level — answer_session_ids points to whole sessions, so retrieval metrics (Recall@k, NDCG@k) are computed by checking whether you retrieved the correct session(s). That makes "one doc per session" the natural starting point for evaluation purposes.

I need to split this into 2 json files: `longmemeval-s-queries` and `longmemeval-s-docs`.

I need the queries and the docs to use the question_id to map to the tenant, so something like this:
  4. Query JSON structure not defined

  The spec says to split into longmemeval-s-queries and longmemeval-s-docs but doesn't define the fields. I'd suggest:

  # longmemeval-s-docs (one entry per session per question)
  {
    "tenant_id": "<question_id>",
    "session_id": "<haystack_session_id>",
    "session_date": "<haystack_date>",
    "session_text": "<concatenated conversation>"
  }

  # longmemeval-s-queries
  {
    "question_id": "<question_id>",
    "question": "<question>",
    "question_date": "<question_date>",
    "answer": "<answer>",
    "answer_session_ids": ["<session_id>", ...],
    "tenant_id": "<question_id>"
  }



--
Let's start by integrating this as a search test, and then later on consider an answer test.