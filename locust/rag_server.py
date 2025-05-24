import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dspy
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'benchmarker', 'src'))

from dspy_rag import VanillaRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vanilla_rag = None
collection_name = "WixKB"
target_property_name = "contents"

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

app = FastAPI(title="Simple VanillaRAG Server")

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global vanilla_rag
    
    logger.info("🚀 Starting up VanillaRAG Server...")
    
    try:
        lm = dspy.LM(
            'openai/gpt-4o', 
            api_key=os.getenv("OPENAI_API_KEY"),
            cache=False,
        )
        dspy.configure(lm=lm)
        
        vanilla_rag = VanillaRAG(collection_name, target_property_name)
        
        logger.info("✅ VanillaRAG initialized and ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Simple query endpoint - send a question, get an answer.
    """
    try:
        if vanilla_rag is None:
            raise HTTPException(status_code=500, detail="RAG instance not initialized")
        
        logger.info(f"🔍 Processing query: {request.question}")
        
        rag_response = vanilla_rag.forward(request.question)
        
        return QueryResponse(answer=rag_response.final_answer)
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)