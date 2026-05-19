"""
Utility module for streaming and preprocessing the Yelp Academic Dataset.

This module provides high-performance generators for reading large JSONL files
without loading them into memory, and utilities for sampling specific subsets 
required for the RAG-based recommendation system.
"""

import json
import os
from typing import Generator, Dict, Any, List, Set


def stream_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Streams a JSONL file line by line to minimize memory footprint.
    
    Args:
        file_path: Absolute or relative path to the .json file.
        
    Yields:
        Dictionary representation of a single record.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_user_review_counts(review_file: str, min_reviews: int = 3) -> Set[str]:
    """
    Identifies users who meet the minimum review threshold for evaluation.
    
    This is a single-pass scan over the review file.
    
    Args:
        review_file: Path to the yelp_academic_dataset_review.json file.
        min_reviews: Minimum number of reviews a user must have.
        
    Returns:
        A set of user_ids that meet the criteria.
    """
    user_counts: Dict[str, int] = {}
    for review in stream_jsonl(review_file):
        u_id = review['user_id']
        user_counts[u_id] = user_counts.get(u_id, 0) + 1
    
    return {u_id for u_id, count in user_counts.items() if count >= min_reviews}


def sample_dataset(
    data_dir: str, 
    output_dir: str, 
    max_evaluation_users: int = 100, 
    max_cold_start_users: int = 50
):
    """
    Creates a working subset of the data for local development and testing.
    
    Samples:
    1. Users with high review counts (for persona extraction).
    2. Users with 0-1 reviews (for cold-start testing).
    3. Associated business metadata for all reviews in the sample.
    
    Args:
        data_dir: Path to raw Yelp dataset files.
        output_dir: Path to save processed JSONL files.
        max_evaluation_users: Limit for high-engagement users.
        max_cold_start_users: Limit for cold-start users.
    """
    review_path = os.path.join(data_dir, 'yelp_academic_dataset_review.json')
    business_path = os.path.join(data_dir, 'yelp_academic_dataset_business.json')
    user_path = os.path.join(data_dir, 'yelp_academic_dataset_user.json')

    # 1. Identify target users
    print("Scanning reviews for user engagement...")
    high_engagement_ids = get_user_review_counts(review_path, min_reviews=3)
    
    # We'll pick a limited set to keep the subset manageable
    target_users_eval = set(list(high_engagement_ids)[:max_evaluation_users])
    
    # 2. Extract reviews for these users
    print(f"Extracting reviews for {len(target_users_eval)} users...")
    sampled_reviews = []
    business_ids_in_sample = set()
    
    for review in stream_jsonl(review_path):
        if review['user_id'] in target_users_eval:
            sampled_reviews.append(review)
            business_ids_in_sample.add(review['business_id'])

    # 3. Extract business metadata for these reviews
    print("Extracting business metadata...")
    sampled_businesses = []
    for biz in stream_jsonl(business_path):
        if biz['business_id'] in business_ids_in_sample:
            sampled_businesses.append(biz)

    # 4. Extract user metadata and collect cold-start user IDs
    print("Extracting user metadata...")
    sampled_user_meta = []
    cold_start_ids = set()
    found_cold_start = 0
    for user in stream_jsonl(user_path):
        u_id = user['user_id']
        if u_id in target_users_eval:
            sampled_user_meta.append(user)
        elif user['review_count'] <= 1 and found_cold_start < max_cold_start_users:
            sampled_user_meta.append(user)
            cold_start_ids.add(u_id)
            found_cold_start += 1

    # Collect any reviews that belong to cold-start users (usually 0 or 1)
    if cold_start_ids:
        print(f"Extracting reviews for {len(cold_start_ids)} cold-start users...")
        for review in stream_jsonl(review_path):
            if review['user_id'] in cold_start_ids:
                sampled_reviews.append(review)
                business_ids_in_sample.add(review['business_id'])

        # Pull any additional businesses surfaced by cold-start reviews
        cold_start_biz_ids = {r['business_id'] for r in sampled_reviews} - {b['business_id'] for b in sampled_businesses}
        if cold_start_biz_ids:
            for biz in stream_jsonl(business_path):
                if biz['business_id'] in cold_start_biz_ids:
                    sampled_businesses.append(biz)

    # 5. Save subsets
    os.makedirs(output_dir, exist_ok=True)
    
    def save_jsonl(data, filename):
        with open(os.path.join(output_dir, filename), 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    print(f"Saving sampled data to {output_dir}...")
    save_jsonl(sampled_reviews, 'reviews_subset.jsonl')
    save_jsonl(sampled_businesses, 'businesses_subset.jsonl')
    save_jsonl(sampled_user_meta, 'users_subset.jsonl')
    print("Sampling complete.")


if __name__ == "__main__":
    # Example usage for testing the loader independently
    RAW_DATA = "data/raw/Yelp-JSON/Yelp JSON/yelp_dataset"
    PROCESSED_DATA = "data/processed"
    
    # Ensure directories exist relative to project root
    if os.path.exists(RAW_DATA):
        sample_dataset(RAW_DATA, PROCESSED_DATA)
    else:
        print(f"Error: Raw data path {RAW_DATA} not found. Run from project root.")
