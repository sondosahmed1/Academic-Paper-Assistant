from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's research question")

class SourceCitation(BaseModel):
    document_id: str
    chunk_id: int
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    context_used: List[str]
