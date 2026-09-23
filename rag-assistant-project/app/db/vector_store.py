import chromadb
from chromadb.config import Settings
import os

class VectorStore:
    """
    Production Vector Store using ChromaDB.
    """
    def __init__(self, persist_directory: str = "C:/ITI_GP/Academic-Paper-Assistant/rag-assistant-project/data/vector_store"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, ids: list, embeddings: list, metadatas: list, documents: list):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query(self, query_embedding: list, n_results: int = 3):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
