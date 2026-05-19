"""
Vector store utility for managing ChromaDB collections.

This module handles the initialization of ChromaDB and the ingestion of 
business and user persona data for semantic retrieval.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class VectorStoreManager:
    """
    Manages ChromaDB collections for businesses and user personas.
    """

    def __init__(self, persist_directory: str = CHROMA_DB_PATH):
        """
        Initializes the ChromaDB client and embedding function.
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )

        self.business_collection = self.client.get_or_create_collection(
            name="businesses",
            embedding_function=self.embedding_function
        )
        
        self.user_collection = self.client.get_or_create_collection(
            name="user_personas",
            embedding_function=self.embedding_function
        )

    def add_businesses(self, businesses: List[Dict[str, Any]]):
        """
        Adds business documents to the vector store.
        
        Expects businesses to have 'business_id', 'name', 'categories', 
        and optionally 'top_reviews' or 'description'.
        """
        ids = []
        documents = []
        metadatas = []
        
        for biz in businesses:
            ids.append(biz['business_id'])
            # Create a rich text representation for embedding
            text = f"Name: {biz['name']}. Categories: {biz.get('categories', '')}. "
            if 'top_reviews' in biz:
                text += f"Reviews: {' '.join(biz['top_reviews'][:3])}"
            
            documents.append(text)
            metadatas.append({
                "name": biz['name'],
                "categories": biz.get('categories', ''),
                "stars": biz.get('stars', 0)
            })

        self.business_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def query_businesses(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves similar businesses based on semantic query.
        """
        results = self.business_collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Format results for easier consumption
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        return formatted

    def add_user_persona(self, user_id: str, persona_text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a user persona summary to the vector store.
        """
        self.user_collection.upsert(
            ids=[user_id],
            documents=[persona_text],
            metadatas=[metadata or {}]
        )

    def get_user_persona(self, user_id: str) -> Optional[str]:
        """
        Retrieves a user's persona text.
        """
        result = self.user_collection.get(ids=[user_id])
        if result['documents']:
            return result['documents'][0]
        return None

if __name__ == "__main__":
    # Quick test
    manager = VectorStoreManager()
    print(f"Collections initialized: {manager.client.list_collections()}")
