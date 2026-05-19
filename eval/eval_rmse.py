"""
Evaluates Task A rating prediction accuracy using leave-one-out protocol.

For each user with 2+ reviews: use all but the last review as history,
predict stars for the held-out business, compare against ground truth.

Outputs RMSE, MAE, and a ±1 star accuracy breakdown.

Run from project root:
    python eval/eval_rmse.py
    python eval/eval_rmse.py --max-users 50
"""

import sys
import os
import argparse
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.review_agent import ReviewAgent
from src.utils.data_loader import stream_jsonl
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = os.getenv("PROCESSED_DATA_DIR", "./data/processed")


def load_business_index(biz_file: str) -> dict:
    index = {}
    for biz in stream_jsonl(biz_file):
        index[biz["business_id"]] = biz
    return index


def load_user_reviews(review_file: str) -> dict:
    user_reviews = defaultdict(list)
    for review in stream_jsonl(review_file):
        user_reviews[review["user_id"]].append(review)
    for uid in user_reviews:
        user_reviews[uid].sort(key=lambda r: r.get("date", ""))
    return user_reviews


def build_history(history_reviews: list, biz_index: dict) -> list:
    history = []
    for r in history_reviews:
        biz = biz_index.get(r["business_id"], {})
        if not biz:
            continue
        history.append({
            "text": r.get("text", ""),
            "stars": r.get("stars", 3),
            "business_name": biz.get("name", "Unknown"),
            "category": biz.get("categories", ""),
        })
    return history


def build_business_metadata(biz: dict) -> dict:
    attrs = biz.get("attributes") or {}
    return {
        "name": biz.get("name", ""),
        "category": biz.get("categories", ""),
        "avg_stars": biz.get("stars", 3.5),
        "price_range": attrs.get("RestaurantsPriceRange2"),
    }


def main():
    parser = argparse.ArgumentParser(description="RMSE eval for Task A star prediction")
    parser.add_argument("--max-users", type=int, default=20,
                        help="Number of users to evaluate. Each user = 1 LLM call.")
    args = parser.parse_args()

    review_file = os.path.join(PROCESSED_DIR, "reviews_subset.jsonl")
    biz_file = os.path.join(PROCESSED_DIR, "businesses_subset.jsonl")

    if not os.path.exists(review_file) or not os.path.exists(biz_file):
        print(f"Error: processed data not found in {PROCESSED_DIR}. Run src/utils/data_loader.py first.")
        sys.exit(1)

    print("Loading data...")
    biz_index = load_business_index(biz_file)
    user_reviews = load_user_reviews(review_file)

    eligible = [(uid, revs) for uid, revs in user_reviews.items() if len(revs) >= 2]
    eval_users = eligible[: args.max_users]
    print(f"Eligible users: {len(eligible)} | Evaluating: {len(eval_users)}\n")

    agent = ReviewAgent()
    squared_errors = []
    abs_errors = []
    skipped = 0

    for uid, reviews in eval_users:
        history_reviews = reviews[:-1]
        target = reviews[-1]

        biz = biz_index.get(target["business_id"])
        if not biz:
            skipped += 1
            continue

        history = build_history(history_reviews, biz_index)
        input_data = {
            "user_id": uid,
            "business_id": target["business_id"],
            "user_history": history,
            "business_metadata": build_business_metadata(biz),
            "nigerian_mode": False,
        }

        try:
            result = agent.run(input_data)
            predicted = float(result.get("predicted_stars", 3))
            actual = float(target["stars"])

            squared_errors.append((predicted - actual) ** 2)
            abs_errors.append(abs(predicted - actual))

            cold = result.get("is_cold_start", False)
            print(f"  {uid[:10]}  actual={actual:.0f}  predicted={predicted:.0f}  "
                  f"error={predicted-actual:+.0f}  cold_start={cold}")

        except Exception as e:
            print(f"  {uid[:10]}  ERROR: {e}")
            skipped += 1

    n = len(squared_errors)
    if n == 0:
        print("\nNo results collected. Check API key and processed data.")
        return

    rmse = math.sqrt(sum(squared_errors) / n)
    mae = sum(abs_errors) / n
    within_half = sum(1 for e in abs_errors if e <= 0.5) / n * 100
    within_one = sum(1 for e in abs_errors if e <= 1.0) / n * 100

    print(f"\n{'=' * 52}")
    print(f"TASK A — RATING ACCURACY  (n={n}, skipped={skipped})")
    print(f"  RMSE              : {rmse:.4f}")
    print(f"  MAE               : {mae:.4f}")
    print(f"  Within ±0.5 stars : {within_half:.1f}%")
    print(f"  Within ±1.0 stars : {within_one:.1f}%")
    print(f"{'=' * 52}")


if __name__ == "__main__":
    main()
