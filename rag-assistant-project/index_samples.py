import os
import logging
import sys

# Add the project root to sys.path so we can import from app.db
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.db.vector_store import VectorStore
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indexer")

def index_sample_data():
    # Initialize the store
    logger.info("Initializing VectorStore...")
    store = VectorStore()

    samples = [
        {
            "text": "Paper 1: Advanced Layout Detection in Academic PDFs. The proposed model achieves an accuracy of 94.5% on the PubLayNet dataset. The primary limitation is the inability to handle multi-column tables with merged cells.",
            "metadata": {"source": "layout_detection.pdf", "page": 1, "region": 0}
        },
        {
            "text": "The authors of the layout detection paper suggest that integrating a transformer-based region proposer will improve the handling of complex tables in future iterations.",
            "metadata": {"source": "layout_detection.pdf", "page": 2, "region": 1}
        },
        {
            "text": "Paper 2: Efficient RAG Systems for Medical Queries. This study introduces a hybrid retrieval method combining BM25 and Dense Vector Search. The system reduced hallucinations by 30% compared to standard RAG.",
            "metadata": {"source": "medical_rag.pdf", "page": 1, "region": 0}
        },
        {
            "text": "The medical RAG system was tested on a corpus of 5,000 PubMed abstracts, demonstrating a significant increase in precision for rare disease queries.",
            "metadata": {"source": "medical_rag.pdf", "page": 3, "region": 2}
        }
    ]

    logger.info(f"Indexing {len(samples)} sample chunks...")

    try:
        # IMPORTANT: Wipe the collection first to avoid duplicate/conflicting embeddings
        try:
            store.client.delete_collection("research_papers")
            logger.info("Deleted old collection to ensure clean state.")
        except:
            pass

        # Re-create the collection
        store.collection = store.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )

        texts = [s["text"] for s in samples]
        metadatas = [s["metadata"] for s in samples]
        ids = [f"sample_{i}" for i in range(len(samples))]

        store.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("Successfully indexed sample papers!")
        print("VERIFICATION: Data has been written to the database.")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")

if __name__ == "__main__":
    index_sample_data()
EOF
