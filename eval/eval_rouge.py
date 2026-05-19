"""
Evaluates Task A review generation quality using ROUGE scores.

Uses the same leave-one-out protocol as eval_rmse.py: holds out each
user's most recent review, generates a replacement, then scores lexical
overlap against the ground truth text.

Requires: pip install rouge-score

Run from project root:
    python eval/eval_rouge.py
    python eval/eval_rouge.py --max-users 20 --nigerian-mode
    python eval/eval_rouge.py --save-examples 5
"""

import sys
import os
import argparse
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.review_agent import ReviewAgent
from src.utils.data_loader import stream_jsonl
from dotenv import load_dotenv

load_dotenv()

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Error: rouge-score not installed.\nRun: pip install rouge-score")
    sys.exit(1)

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
    parser = argparse.ArgumentParser(description="ROUGE eval for Task A review generation")
    parser.add_argument("--max-users", type=int, default=20,
                        help="Number of users to evaluate. Each user = 1 LLM call.")
    parser.add_argument("--nigerian-mode", action="store_true",
                        help="Run with nigerian_mode=True. Useful for comparing mode on/off.")
    parser.add_argument("--save-examples", type=int, default=0,
                        help="Save N example (ground_truth, generated) pairs to eval/examples.jsonl")
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
    print(f"Eligible users: {len(eligible)} | Evaluating: {len(eval_users)} | "
          f"nigerian_mode={args.nigerian_mode}\n")

    agent = ReviewAgent()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_scores, r2_scores, rL_scores = [], [], []
    examples = []
    skipped = 0

    for uid, reviews in eval_users:
        history_reviews = reviews[:-1]
        target = reviews[-1]

        biz = biz_index.get(target["business_id"])
        ground_truth = target.get("text", "").strip()

        if not biz or not ground_truth:
            skipped += 1
            continue

        history = build_history(history_reviews, biz_index)
        input_data = {
            "user_id": uid,
            "business_id": target["business_id"],
            "user_history": history,
            "business_metadata": build_business_metadata(biz),
            "nigerian_mode": args.nigerian_mode,
        }

        try:
            result = agent.run(input_data)
            generated = result.get("generated_review", "").strip()

            if not generated:
                print(f"  {uid[:10]}  EMPTY generated review — skipping")
                skipped += 1
                continue

            scores = scorer.score(ground_truth, generated)
            r1 = scores["rouge1"].fmeasure
            r2 = scores["rouge2"].fmeasure
            rL = scores["rougeL"].fmeasure

            r1_scores.append(r1)
            r2_scores.append(r2)
            rL_scores.append(rL)

            print(f"  {uid[:10]}  R1={r1:.3f}  R2={r2:.3f}  RL={rL:.3f}  "
                  f"biz={biz.get('name', '')[:30]}")

            if args.save_examples and len(examples) < args.save_examples:
                examples.append({
                    "user_id": uid,
                    "business": biz.get("name", ""),
                    "actual_stars": target["stars"],
                    "predicted_stars": result.get("predicted_stars"),
                    "ground_truth": ground_truth,
                    "generated": generated,
                    "rouge1": round(r1, 4),
                    "rougeL": round(rL, 4),
                    "nigerian_mode": args.nigerian_mode,
                })

        except Exception as e:
            print(f"  {uid[:10]}  ERROR: {e}")
            skipped += 1

    n = len(r1_scores)
    if n == 0:
        print("\nNo results collected. Check API key and processed data.")
        return

    print(f"\n{'=' * 52}")
    print(f"TASK A — REVIEW QUALITY  (n={n}, skipped={skipped})")
    print(f"  ROUGE-1 F1 : {sum(r1_scores)/n:.4f}")
    print(f"  ROUGE-2 F1 : {sum(r2_scores)/n:.4f}")
    print(f"  ROUGE-L F1 : {sum(rL_scores)/n:.4f}")
    print(f"{'=' * 52}")
    print("Note: ROUGE measures lexical overlap against ground truth reviews.")
    print("Generated reviews won't match verbatim — scores of 0.10-0.25 are")
    print("normal and competitive for open-ended review generation tasks.")
    print("Behavioural fidelity (tone, rating pattern) is the primary signal.")

    if examples:
        out_path = os.path.join(os.path.dirname(__file__), "examples.jsonl")
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"\nSaved {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
