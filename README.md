# RAG-Powered Document Assistant

A local academic/research-paper assistant that retrieves semantically relevant PDF passages, adds detected page-layout context, and asks a local Ollama model to return a grounded, cited answer.

## Architecture

```text
PDF corpus -> PyMuPDF extraction -> 500-token chunks -> MiniLM embeddings -> ChromaDB
       |                                                        |
       +-> page images -> LayoutParser/PubLayNet -> layout_context.json
                                                                |
Streamlit UI -> FastAPI /query -> retrieve top-k + layout context -> Ollama -> cited answer
```

## Tech stack

Python 3.10, FastAPI/Uvicorn, Streamlit, ChromaDB, `sentence-transformers/all-MiniLM-L6-v2`, Ollama (`llama3.1:8b`, with `phi3:mini` fallback), PyMuPDF, and LayoutParser/PubLayNet.

## Project structure

```text
rag-assistant-project/
├── notebooks/rag_pipeline.ipynb
├── backend/                 # API, Chroma persistence, tests, Dockerfile
└── frontend/                # Streamlit chat client
```

## Domain and data

Use a coherent collection of text-extractable academic PDFs (for example, arXiv papers about transformers). The notebook reports document/page counts and parse failures. It renders each page for layout detection and stores a concise per-page description such as `Page 4 contains: 1 figure, 1 table` beside the vector store. Layout context is presented to the LLM, not treated as visual understanding.

## Setup

1. Install Python 3.10 and [Ollama](https://ollama.com/), then run `ollama pull llama3.1:8b` (or set `OLLAMA_MODEL=phi3:mini`).
2. Place PDFs in `backend/data/corpus/`.
3. Create and activate a virtual environment. Install `pip install -r backend/requirements.txt`.
4. Copy `backend/.env.example` to `backend/.env` and run the notebook top-to-bottom to create `backend/data/vector_store/`.
5. Start the API from `backend`: `uvicorn app.main:app --reload --port 8000`.
6. In another terminal: `pip install -r frontend/requirements.txt` then `streamlit run frontend/app.py`.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Generation model |
| `CHROMA_COLLECTION` | `academic_papers` | Persisted Chroma collection |
| `CHROMA_PATH` | `./data/vector_store` | Chroma and layout metadata location |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per request |
| `CORS_ORIGINS` | `http://localhost:8501` | Allowed Streamlit origin |
| `API_BASE_URL` | `http://localhost:8000` | Frontend API URL |

## API

`GET /health` confirms the API and configured model. `POST /query` accepts `{ "question": "..." }` and returns `{ "answer": "...", "sources": ["..."] }`.

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"What does the paper say about self-attention?\"}"
```

## Evaluation results

Run the ten-question evaluation table in the notebook after indexing the selected corpus. Record each question, retrieved source, answer, and correctness. Typical failure cases include ambiguous terminology and insufficient context; mitigation is a coherent corpus, overlap between chunks, top-k retrieval, and explicit abstention when sources do not support an answer.

## Screenshots

_Add screenshots of the Streamlit conversation, `/docs`, and a notebook layout-detection result here._