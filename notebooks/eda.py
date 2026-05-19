import json
import pandas as pd
from collections import Counter

# Sample first 100k reviews only
reviews = []
with open("/home/kaftandev/Keuro Lab/bct-hackathon/data/raw/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json", "r") as f:
    for i, line in enumerate(f):
        if i >= 100000:
            break
        reviews.append(json.loads(line))

df_reviews = pd.DataFrame(reviews)

print("=== REVIEWS ===")
print(f"Columns: {df_reviews.columns.tolist()}")
print(f"Shape: {df_reviews.shape}")
print(f"Star distribution:\n{df_reviews['stars'].value_counts().sort_index()}")
print(f"Avg review length: {df_reviews['text'].str.len().mean():.0f} chars")
print(f"Unique users: {df_reviews['user_id'].nunique()}")
print(f"Unique businesses: {df_reviews['business_id'].nunique()}")

# Reviews per user distribution
reviews_per_user = df_reviews['user_id'].value_counts()
print(f"\n=== USER ACTIVITY (in sample) ===")
print(f"Users with 1 review: {(reviews_per_user == 1).sum()}")
print(f"Users with 2-5 reviews: {((reviews_per_user >= 2) & (reviews_per_user <= 5)).sum()}")
print(f"Users with 6+ reviews: {(reviews_per_user >= 6).sum()}")

# Business sample
businesses = []
with open("yelp_academic_dataset_business.json", "r") as f:
    for line in f:
        businesses.append(json.loads(line))

df_biz = pd.DataFrame(businesses)
print(f"\n=== BUSINESSES ===")
print(f"Columns: {df_biz.columns.tolist()}")
print(f"Total businesses: {len(df_biz)}")
print(f"Top categories sample:")

# Extract categories
all_cats = []
for cats in df_biz['categories'].dropna():
    all_cats.extend([c.strip() for c in cats.split(',')])
cat_counts = Counter(all_cats)
print(pd.Series(cat_counts).sort_values(ascending=False).head(20))