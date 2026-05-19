"""
Ingestion script for the Yelp RAG system.

Reads processed JSONL files and populates the ChromaDB vector store.
"""

import os
import json
from src.utils.vector_store import VectorStoreManager
from src.utils.data_loader import stream_jsonl
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "./data/processed")

def ingest():
    """
    Ingests sampled businesses and reviews into ChromaDB.
    """
    manager = VectorStoreManager()
    
    business_file = os.path.join(PROCESSED_DATA_DIR, 'businesses_subset.jsonl')
    review_file = os.path.join(PROCESSED_DATA_DIR, 'reviews_subset.jsonl')
    
    if not os.path.exists(business_file):
        print(f"Error: {business_file} not found. Run data_loader.py first.")
        return

    # 1. Collect top reviews for each business to enrich metadata
    print("Collecting review snippets for business enrichment...")
    biz_reviews = {}
    for review in stream_jsonl(review_file):
        b_id = review['business_id']
        if b_id not in biz_reviews:
            biz_reviews[b_id] = []
        if len(biz_reviews[b_id]) < 3: # Keep top 3 snippets
            biz_reviews[b_id].append(review['text'][:200])

    # 2. Load and ingest businesses
    print("Ingesting businesses...")
    businesses = []
    for biz in stream_jsonl(business_file):
        biz['top_reviews'] = biz_reviews.get(biz['business_id'], [])
        businesses.append(biz)
        
        # Batch ingestion
        if len(businesses) >= 100:
            manager.add_businesses(businesses)
            businesses = []
            
    if businesses:
        manager.add_businesses(businesses)
        
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
