"""
Data ingestion and aggregation pipeline for newborn/baby product research.

This module expects local CSV exports (no live scraping or unofficial APIs).
The default filenames are:
- social_posts.csv
- products_catalog.csv
- image_matches.csv (optional)

Run `python data_pipeline.py` to generate `aggregated_products.csv`.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils_text import clean_text, contains_keywords, similarity

# Configurable keyword lists and thresholds.
BABY_KEYWORDS = [
    "baby",
    "newborn",
    "stroller",
    "pram",
    "crib",
    "cot",
    "diaper",
    "nappy",
    "onesie",
    "bottle",
    "pacifier",
    "car seat",
    "carrier",
    "swaddle",
    "wipes",
]
CATEGORY_KEYWORDS = {
    "stroller": ["stroller", "pram"],
    "crib": ["crib", "cot"],
    "diaper": ["diaper", "nappy", "wipes"],
    "bottle": ["bottle", "pacifier"],
    "car seat": ["car seat"],
    "carrier": ["carrier"],
    "onesie": ["onesie", "bodysuit"],
}
TEXT_MATCH_THRESHOLD = 60.0  # similarity score between 0-100
RECENT_DAYS = 90


class DataValidationError(Exception):
    """Raised when required columns are missing."""


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str, required_cols: Sequence[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise DataValidationError(f"{path} missing columns: {missing}")
    return df


def load_social_posts(path: str = "social_posts.csv") -> pd.DataFrame:
    required = [
        "post_id",
        "image_id",
        "caption",
        "hashtags",
        "likes",
        "comments",
        "posted_at",
        "platform",
    ]
    df = _load_csv(path, required)
    return df


def load_products_catalog(path: str = "products_catalog.csv") -> pd.DataFrame:
    required = [
        "product_id",
        "product_name",
        "brand",
        "model",
        "category",
        "price",
        "currency",
        "url",
        "rating",
        "marketplace",
    ]
    df = _load_csv(path, required)
    return df


def load_image_matches(path: str = "image_matches.csv") -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    required = ["image_id", "product_id", "score"]
    df = _load_csv(path, required)
    return df


# ---------------------------------------------------------------------------
# Cleaning and enrichment
# ---------------------------------------------------------------------------

def clean_posts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["caption", "hashtags", "platform"]:
        df[col] = df[col].fillna("").astype(str)
    df["caption_clean"] = df["caption"].apply(clean_text)
    df["hashtags_clean"] = df["hashtags"].apply(clean_text)
    df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce")
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df["comments"] = pd.to_numeric(df["comments"], errors="coerce").fillna(0).astype(int)
    return df


def filter_baby_posts(df: pd.DataFrame, keywords: Sequence[str] = BABY_KEYWORDS) -> pd.DataFrame:
    df = df.copy()
    df["is_baby_related"] = df.apply(
        lambda row: contains_keywords(f"{row['caption']} {row['hashtags']}", keywords), axis=1
    )
    return df[df["is_baby_related"]]


def infer_category(text: str) -> Optional[str]:
    for category, kw_list in CATEGORY_KEYWORDS.items():
        if contains_keywords(text, kw_list):
            return category
    return None


def infer_brand_model(text: str, products: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    brand_scores: Dict[str, float] = {}
    model_scores: Dict[str, float] = {}
    for brand in products["brand"].dropna().unique():
        brand_scores[brand] = similarity(text, brand)
    for model in products["model"].dropna().unique():
        model_scores[model] = similarity(text, model)
    best_brand = max(brand_scores.items(), key=lambda x: x[1], default=(None, 0))[0]
    best_model = max(model_scores.items(), key=lambda x: x[1], default=(None, 0))[0]
    return best_brand, best_model


def annotate_posts(df_posts: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = df_posts.copy()
    combined_text = df.apply(lambda r: f"{r['caption']} {r['hashtags']}", axis=1)
    df["inferred_category"] = combined_text.apply(infer_category)
    inferred_brands = []
    inferred_models = []
    for text in combined_text:
        brand, model = infer_brand_model(text, products)
        inferred_brands.append(brand)
        inferred_models.append(model)
    df["inferred_brand"] = inferred_brands
    df["inferred_model"] = inferred_models
    return df


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def apply_image_matches(posts: pd.DataFrame, image_matches: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = posts.copy()
    if image_matches is None or image_matches.empty:
        df["matched_product_id"] = None
        df["match_method"] = None
        return df
    image_matches_sorted = image_matches.sort_values(by="score", ascending=False)
    best_matches = image_matches_sorted.drop_duplicates(subset=["image_id"])
    df = df.merge(best_matches[["image_id", "product_id", "score"]], on="image_id", how="left")
    df = df.rename(columns={"product_id": "matched_product_id", "score": "image_match_score"})
    df["match_method"] = df["matched_product_id"].apply(lambda x: "image" if pd.notna(x) else None)
    return df


def text_match_posts(
    posts: pd.DataFrame, products: pd.DataFrame, threshold: float = TEXT_MATCH_THRESHOLD
) -> pd.DataFrame:
    df = posts.copy()
    unmatched_mask = df["matched_product_id"].isna()
    product_texts = products.assign(
        combined_text=lambda d: d[["product_name", "brand", "model", "category"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )
    for idx in df[unmatched_mask].index:
        text = f"{df.at[idx, 'caption']} {df.at[idx, 'hashtags']}"
        scores: List[Tuple[str, float]] = []
        for _, prod in product_texts.iterrows():
            score = similarity(text, prod["combined_text"])
            scores.append((prod["product_id"], score))
        if not scores:
            continue
        best_product, best_score = max(scores, key=lambda x: x[1])
        if best_score >= threshold:
            df.at[idx, "matched_product_id"] = best_product
            df.at[idx, "match_method"] = "text"
            df.at[idx, "text_match_score"] = best_score
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_products(posts: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    recent_cutoff = datetime.utcnow() - timedelta(days=RECENT_DAYS)
    posts_with_product = posts[pd.notna(posts["matched_product_id"])]
    if posts_with_product.empty:
        return pd.DataFrame()

    agg = posts_with_product.groupby("matched_product_id").agg(
        num_posts=("post_id", "count"),
        total_likes=("likes", "sum"),
        total_comments=("comments", "sum"),
        avg_likes=("likes", "mean"),
        avg_comments=("comments", "mean"),
        recent_post_count=("posted_at", lambda s: (s >= recent_cutoff).sum()),
    )

    prices = products.groupby("product_id").agg(
        min_price=("price", "min"),
        max_price=("price", "max"),
        median_price=("price", "median"),
        avg_price=("price", "mean"),
        currency=("currency", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
    )

    def collect_urls(group: pd.DataFrame) -> List[str]:
        urls = group.dropna(subset=["url"])["url"].drop_duplicates().head(3).tolist()
        return urls

    url_examples = products.groupby("product_id").apply(collect_urls)
    url_df = url_examples.apply(pd.Series)
    url_df.columns = [f"price_url_{i+1}" for i in range(url_df.shape[1])]

    merged = products.drop_duplicates(subset=["product_id"]).set_index("product_id")
    merged = merged.join(agg, how="inner")
    merged = merged.join(prices, how="left", rsuffix="_price")
    merged = merged.join(url_df, how="left")
    merged = merged.reset_index()
    return merged


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    social_path: str = "social_posts.csv",
    products_path: str = "products_catalog.csv",
    image_matches_path: str = "image_matches.csv",
    output_path: str = "aggregated_products.csv",
) -> pd.DataFrame:
    posts = load_social_posts(social_path)
    products = load_products_catalog(products_path)
    image_matches = load_image_matches(image_matches_path)

    posts = clean_posts(posts)
    posts = filter_baby_posts(posts)
    posts = annotate_posts(posts, products)
    posts = apply_image_matches(posts, image_matches)
    posts = text_match_posts(posts, products)

    aggregated = aggregate_products(posts, products)
    aggregated.to_csv(output_path, index=False)
    return aggregated


def describe_expected_schema() -> Dict[str, List[str]]:
    return {
        "social_posts.csv": [
            "post_id",
            "image_id",
            "image_url",
            "caption",
            "hashtags",
            "likes",
            "comments",
            "posted_at",
            "platform",
        ],
        "products_catalog.csv": [
            "product_id",
            "product_name",
            "brand",
            "model",
            "category",
            "price",
            "currency",
            "url",
            "rating",
            "marketplace",
        ],
        "image_matches.csv (optional)": ["image_id", "product_id", "score"],
    }


if __name__ == "__main__":
    try:
        result = run_pipeline()
        print("Aggregated data saved to aggregated_products.csv")
        print(json.dumps(describe_expected_schema(), indent=2))
    except (FileNotFoundError, DataValidationError) as exc:
        print(f"Error: {exc}")
