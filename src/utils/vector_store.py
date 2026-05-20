"""
Vector store utility — ChromaDB (local/Docker) or Pinecone (cloud/Render).

Backend selected automatically:
  PINECONE_API_KEY set  →  Pinecone
  otherwise             →  ChromaDB (default)
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bct-hackathon")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

_BACKEND = "pinecone" if PINECONE_API_KEY else "chroma"


class VectorStoreManager:

    def __init__(self, persist_directory: str = CHROMA_DB_PATH):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self._backend = _BACKEND

        if self._backend == "pinecone":
            self._init_pinecone()
        else:
            self._init_chroma(persist_directory)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_chroma(self, persist_directory: str):
        import chromadb
        from chromadb.utils import embedding_functions

        self._client = chromadb.PersistentClient(path=persist_directory)
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        self.business_collection = self._client.get_or_create_collection(
            name="businesses", embedding_function=ef
        )
        self.user_collection = self._client.get_or_create_collection(
            name="user_personas", embedding_function=ef
        )

    def _init_pinecone(self):
        from pinecone import Pinecone, ServerlessSpec
        from openai import OpenAI

        self._openai = OpenAI(api_key=OPENAI_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)

        existing = [i.name for i in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self._index = pc.Index(PINECONE_INDEX_NAME)

    # ------------------------------------------------------------------
    # Embed helper (Pinecone path only)
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        resp = self._openai.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return resp.data[0].embedding

    # ------------------------------------------------------------------
    # Public API — same interface regardless of backend
    # ------------------------------------------------------------------

    def add_businesses(self, businesses: List[Dict[str, Any]]):
        ids, documents, metadatas = [], [], []

        for biz in businesses:
            ids.append(biz["business_id"])
            text = f"Name: {biz['name']}. Categories: {biz.get('categories', '')}. "
            if "top_reviews" in biz:
                text += f"Reviews: {' '.join(biz['top_reviews'][:3])}"
            documents.append(text)
            metadatas.append({
                "name": biz["name"],
                "categories": biz.get("categories", ""),
                "stars": biz.get("stars", 0),
            })

        if self._backend == "pinecone":
            vectors = []
            for i, doc in enumerate(documents):
                vectors.append({
                    "id": f"biz_{ids[i]}",
                    "values": self._embed(doc),
                    "metadata": {**metadatas[i], "doc": doc, "business_id": ids[i]},
                })
            # Pinecone upsert in batches of 100
            for start in range(0, len(vectors), 100):
                self._index.upsert(vectors=vectors[start:start + 100], namespace="businesses")
        else:
            self.business_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query_businesses(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if self._backend == "pinecone":
            vec = self._embed(query_text)
            res = self._index.query(
                vector=vec,
                top_k=n_results,
                namespace="businesses",
                include_metadata=True,
            )
            return [
                {
                    "id": m.metadata.get("business_id", m.id),
                    "document": m.metadata.get("doc", ""),
                    "metadata": {
                        "name": m.metadata.get("name", ""),
                        "categories": m.metadata.get("categories", ""),
                        "stars": m.metadata.get("stars", 0),
                    },
                    "distance": 1 - m.score,
                }
                for m in res.matches
            ]
        else:
            results = self.business_collection.query(
                query_texts=[query_text], n_results=n_results
            )
            return [
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
                for i in range(len(results["ids"][0]))
            ]

    def add_user_persona(self, user_id: str, persona_text: str, metadata: Optional[Dict[str, Any]] = None):
        if self._backend == "pinecone":
            self._index.upsert(
                vectors=[{
                    "id": f"user_{user_id}",
                    "values": self._embed(persona_text),
                    "metadata": {**(metadata or {}), "doc": persona_text, "user_id": user_id},
                }],
                namespace="users",
            )
        else:
            self.user_collection.upsert(
                ids=[user_id], documents=[persona_text], metadatas=[metadata or {}]
            )

    def get_user_persona(self, user_id: str) -> Optional[str]:
        if self._backend == "pinecone":
            res = self._index.fetch(ids=[f"user_{user_id}"], namespace="users")
            vec = res.vectors.get(f"user_{user_id}")
            return vec.metadata.get("doc") if vec else None
        else:
            result = self.user_collection.get(ids=[user_id])
            return result["documents"][0] if result["documents"] else None


if __name__ == "__main__":
    manager = VectorStoreManager()
    print(f"Backend: {manager._backend}")
