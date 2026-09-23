import logging
import requests
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import query
from app.db.vector_store import VectorStore
from app.core.layout_detector import LayoutDetector
from app.services.rag_service import RAGService
from app.services.ollama_client import OllamaClient

# Production logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan: Loads expensive resources ONCE at startup.
    """
    logger.info("Starting up: Loading Vector Store and LLM...")

    # 1. Initialize Resource-Heavy Components
    vector_store = VectorStore()
    layout_detector = LayoutDetector()
    llm_client = OllamaClient()

    # --- STARTUP VALIDATION ---
    try:
        # Health Check
        health_res = requests.get(llm_client.base_url.rstrip('/'), timeout=5)
        if health_res.status_code != 200:
            logger.warning(f"Ollama health check failed with status {health_res.status_code}")
        else:
            logger.info("Ollama connectivity verified.")

        # Model Check
        tags_res = requests.get(f"{llm_client.base_url.rstrip('/')}/api/tags", timeout=5)
        tags_res.raise_for_status()
        pulled_models = [tag['name'] for tag in tags_res.json().get('models', [])]

        if llm_client.model not in pulled_models:
            logger.error(f"CRITICAL: Model '{llm_client.model}' not found in Ollama. Please run: ollama pull {llm_client.model}")
        else:
            logger.info(f"Model '{llm_client.model}' verified and ready.")

    except Exception as e:
        logger.error(f"Ollama Startup Validation Error: {e}. Is 'ollama serve' running?")

    # 2. Initialize RAG Service
    rag_service = RAGService(vector_store, layout_detector, llm_client)

    # 3. Attach to app state for dependency injection
    app.state.rag_service = rag_service

    logger.info("System Ready.")
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Register Routes
app.include_router(query.router, prefix="/api", tags=["Query"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
