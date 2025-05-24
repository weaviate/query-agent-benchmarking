import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import weaviate
from weaviate.agents.query import AsyncQueryAgent
from weaviate.auth import Auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

query_agent = None
weaviate_client = None
COLLECTION = "WixKB"

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list = []

app = FastAPI(title="Weaviate Async Query Agent Server")

@app.on_event("startup")
async def startup_event():
    global query_agent, weaviate_client
    
    logger.info("Starting up Weaviate Async Query Agent Server...")
    
    try:
        weaviate_client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
            headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
        )
        await weaviate_client.connect()
        
        query_agent = AsyncQueryAgent(
            client=weaviate_client,
            collections=[COLLECTION],
            agents_host="https://api.agents.weaviate.io"
        )
        
        logger.info("Weaviate Async Query Agent initialized and ready!")
        logger.info(f"Using collection: {COLLECTION}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Async Query Agent: {e}")
        raise

@app.post("/query", response_model=QueryResponse)
async def query_agent_endpoint(request: QueryRequest):
    try:
        global query_agent
        
        if query_agent is None:
            raise HTTPException(status_code=500, detail="Async Query Agent not initialized")
        
        logger.info(f"Processing query: {request.question}")
        
        agent_response = await query_agent.run(request.question)
        
        sources = []
        if hasattr(agent_response, 'sources') and agent_response.sources:
            sources = agent_response.sources
        
        return QueryResponse(
            answer=agent_response.final_answer,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Weaviate Async Query Agent Server",
        "collection": COLLECTION
    }

@app.on_event("shutdown")
async def shutdown_event():
    global weaviate_client
    if weaviate_client:
        await weaviate_client.close()
        logger.info("Weaviate client connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)