from typing import List, Tuple
import logging
from app.db.vector_store import VectorStore
from app.core.layout_detector import LayoutDetector

logger = logging.getLogger("rag_service")

class RAGService:
    """
    Core RAG logic: Retrieval + Grounded Generation.
    """
    def __init__(self, store: VectorStore, layout_detector: LayoutDetector, llm_client):
        self.store = store
        self.layout_detector = layout_detector
        self.llm = llm_client

    def retrieve(self, query_embedding: list) -> List[Tuple[str, str, str]]:
        """Retrieves chunks and their layout context using a pre-computed embedding."""
        results = self.store.query(query_embedding)

        retrieved_chunks = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                # Use CV component to get context
                layout_type = self.layout_detector.get_page_context(
                    meta.get("page", 0), meta.get("region", 0)
                )
                retrieved_chunks.append((doc, meta['source'], layout_type))

        return retrieved_chunks

    def retrieve_text(self, query_text: str) -> List[Tuple[str, str, str]]:
        """Retrieves chunks and their layout context using raw text (lets ChromaDB handle embedding)."""
        logger.info(f"Retrieving text for query: {query_text}")

        # Use the store's internal collection to query by text
        results = self.store.collection.query(
            query_texts=[query_text],
            n_results=3
        )

        logger.info(f"ChromaDB results: {results}")

        retrieved_chunks = []
        if results and results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                # Use CV component to get context
                layout_type = self.layout_detector.get_page_context(
                    meta.get("page", 0), meta.get("region", 0)
                )
                retrieved_chunks.append((doc, meta.get('source', 'unknown'), layout_type))

        logger.info(f"Returning {len(retrieved_chunks)} chunks")
        return retrieved_chunks

    def generate_answer(self, query: str, context_chunks: List[Tuple[str, str, str]]) -> Tuple[str, List]:
        """
        Grounded Generation: Strictly citations, no external knowledge.
        """
        if not context_chunks:
            return "I couldn't find any relevant information in the indexed research papers to answer this question.", []

        # Build the grounded prompt
        context_text = "\n\n".join([
            f"[Source: {src}] (Type: {layout}) {text}"
            for text, src, layout in context_chunks
        ])

        prompt = f"""You are a professional research assistant. Your goal is to provide a highly accurate answer based ONLY on the provided context.

CONSTRAINTS:
1. Use ONLY the provided CONTEXT. Do not use external knowledge.
2. If the answer is not contained within the context, explicitly state: "I don't know based on the provided papers."
3. Every factual claim MUST be cited using the format [Source: filename].
4. Maintain a professional, academic tone.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

        # Real LLM call via the injected client
        answer = self.llm.complete(prompt)

        # Extract citations for the schema (passing full text as chunk_id for simplicity in this version)
        citations = [{"document_id": src, "chunk_id": "0", "text": text} for text, src, layout in context_chunks]

        return answer, citations
