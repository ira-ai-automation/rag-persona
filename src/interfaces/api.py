"""FastAPI REST API for Local RAG Assistant."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.pipeline import RAGPipeline
from src.utils.config import load_config


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    max_sources: Optional[int] = 5
    temperature: Optional[float] = None


class Source(BaseModel):
    title: str
    path: str
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    query: str
    processing_time: float


class StatusResponse(BaseModel):
    status: str
    documents_indexed: int
    pipeline_ready: bool


# Initialize FastAPI app
app = FastAPI(
    title="Local RAG Assistant API",
    description="REST API for offline document Q&A system",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup."""
    global pipeline
    try:
        config = load_config()
        pipeline = RAGPipeline(config)
        pipeline.initialize_pipeline()
        print("✅ RAG Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        pipeline = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "Local RAG Assistant API", "status": "running"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status."""
    if pipeline is None:
        return StatusResponse(
            status="error",
            documents_indexed=0,
            pipeline_ready=False
        )
    
    try:
        doc_count = pipeline.retriever.get_document_count()
        return StatusResponse(
            status="ready",
            documents_indexed=doc_count,
            pipeline_ready=True
        )
    except Exception as e:
        return StatusResponse(
            status="error",
            documents_indexed=0,
            pipeline_ready=False
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        import time
        start_time = time.time()
        
        # Process query
        response = pipeline.query(request.question)
        
        processing_time = time.time() - start_time
        
        # Format sources
        sources = []
        for source in response.get("sources", []):
            sources.append(Source(
                title=source.get("title", "Unknown"),
                path=source.get("path", ""),
                content=source.get("content", ""),
                score=source.get("score", 0.0)
            ))
        
        return QueryResponse(
            answer=response.get("answer", "No answer generated"),
            sources=sources[:request.max_sources],
            query=request.question,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/documents/count")
async def get_document_count():
    """Get the number of indexed documents."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        count = pipeline.retriever.get_document_count()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document count: {str(e)}")


def main():
    """Run the API server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
