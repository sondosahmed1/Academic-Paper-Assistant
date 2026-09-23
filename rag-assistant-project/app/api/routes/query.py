from fastapi import APIRouter, HTTPException, Depends, Request
from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_service import RAGService
from typing import Annotated
import logging

logger = logging.getLogger("api")

router = APIRouter()

# Dependency to get the RAG service from app state
def get_rag_service(request: Request):
    return request.app.state.rag_service

@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: Request,
    rag_service: Annotated[RAGService, Depends(get_rag_service)] = None
):
    """
    Handles natural language queries.
    Manual parsing of the body to eliminate Pydantic 422 wrapper issues.
    """
    try:
        # Manually extract the JSON body
        body = await request.json()

        # The frontend sends {"query": "..."}. We ensure it's here.
        query_text = body.get("query")

        if not query_text:
            raise HTTPException(status_code=422, detail="Field 'query' is missing or empty in request body")

        # 1. Generate query embedding
        # FIX: Use the vector store's default embedding function instead of hardcoded [0.1] * 384
        # Since we are using ChromaDB's default embedding function in index_samples.py,
        # we should pass the text query directly to rag_service.retrieve.

        # 2. Retrieve and Generate
        # We modify retrieve to handle the text query directly
        answer, citations = rag_service.generate_answer(
            query_text,
            rag_service.retrieve_text(query_text)
        )

        return QueryResponse(
            answer=answer,
            sources=citations,
            context_used=[]
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
