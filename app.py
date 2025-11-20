"""
Streamlit dashboard for visualizing aggregated newborn/baby product research.

Usage:
    python data_pipeline.py   # generate aggregated_products.csv
    streamlit run app.py      # launch dashboard

The app only reads local CSV files; no live scraping or unofficial API calls.
"""
from __future__ import annotations

import os
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from data_pipeline import BABY_KEYWORDS, clean_posts, filter_baby_posts

AGGREGATED_PATH = "aggregated_products.csv"
SOCIAL_POSTS_PATH = "social_posts.csv"


def load_data(path: str = AGGREGATED_PATH) -> pd.DataFrame:
    """Load aggregated product data, stopping the app if missing."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(
            "aggregated_products.csv not found. Run `python data_pipeline.py` to generate it."
        )
        st.stop()
    return df


def load_posts_for_time_series(path: str = SOCIAL_POSTS_PATH) -> Optional[pd.DataFrame]:
    """Load and filter raw social posts for optional time-series chart."""
    if not os.path.exists(path):
        return None
    posts = pd.read_csv(path)
    if {"caption", "hashtags", "posted_at"}.issubset(posts.columns):
        posts = clean_posts(posts)
        posts = filter_baby_posts(posts, keywords=BABY_KEYWORDS)
        return posts
    return None


def sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")
    categories = sorted(df["category"].dropna().unique())
    brands = sorted(df["brand"].dropna().unique())

    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
    selected_brands = st.sidebar.multiselect("Brand", brands, default=brands)

    price_min = float(df["min_price"].min()) if not df.empty else 0.0
    price_max = float(df["max_price"].max()) if not df.empty else 0.0
    price_range = st.sidebar.slider(
        "Price range", min_value=price_min, max_value=price_max, value=(price_min, price_max)
    )

    min_posts = int(st.sidebar.number_input("Minimum number of posts", min_value=0, value=0))
    min_likes = int(st.sidebar.number_input("Minimum total likes", min_value=0, value=0))

    return {
        "categories": selected_categories,
        "brands": selected_brands,
        "price_range": price_range,
        "min_posts": min_posts,
        "min_likes": min_likes,
    }


def apply_filters(df: pd.DataFrame, filters):
    mask = (
        df["category"].isin(filters["categories"])
        & df["brand"].isin(filters["brands"])
        & df["median_price"].between(filters["price_range"][0], filters["price_range"][1])
        & (df["num_posts"] >= filters["min_posts"])
        & (df["total_likes"] >= filters["min_likes"])
    )
    return df[mask]


def render_kpis(df: pd.DataFrame, posts_total: int):
    st.subheader("High-level KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Distinct products", f"{df['product_id'].nunique():,}")
    c2.metric("Total posts considered", f"{posts_total:,}")
    top_categories = (
        df.groupby("category")["num_posts"].sum().sort_values(ascending=False).head(5)
    )
    top_cat_str = ", ".join([f"{cat} ({cnt})" for cat, cnt in top_categories.items()])
    c3.metric("Top categories (posts)", top_cat_str or "N/A")


def render_top_products_table(df: pd.DataFrame):
    st.subheader("Top products")
    if df.empty:
        st.info("No products match the current filters.")
        return

    display_df = df.copy()
    url_cols = [col for col in df.columns if col.startswith("price_url_")]
    for col in url_cols:
        display_df[col] = display_df[col].fillna("").apply(
            lambda x: f"[link]({x})" if isinstance(x, str) and x else ""
        )

    display_df = display_df[
        [
            "product_name",
            "brand",
            "model",
            "category",
            "num_posts",
            "total_likes",
            "avg_likes",
            "median_price",
            "currency",
            *url_cols,
        ]
    ]
    st.dataframe(
        display_df.sort_values(by="num_posts", ascending=False),
        use_container_width=True,
    )


def chart_top_categories(df: pd.DataFrame):
    top_cat = df.groupby("category")["num_posts"].sum().reset_index()
    top_cat = top_cat.sort_values("num_posts", ascending=False).head(10)
    chart = (
        alt.Chart(top_cat)
        .mark_bar()
        .encode(x=alt.X("num_posts", title="Number of posts"), y=alt.Y("category", sort="-x"))
    )
    st.altair_chart(chart, use_container_width=True)


def chart_top_brands(df: pd.DataFrame):
    top_brands = df.groupby("brand")["num_posts"].sum().reset_index()
    top_brands = top_brands.sort_values("num_posts", ascending=False).head(10)
    chart = (
        alt.Chart(top_brands)
        .mark_bar()
        .encode(x=alt.X("num_posts", title="Number of posts"), y=alt.Y("brand", sort="-x"))
    )
    st.altair_chart(chart, use_container_width=True)


def chart_time_series(posts: Optional[pd.DataFrame]):
    st.subheader("Posting trend (baby-related posts)")
    if posts is None:
        st.info("Load social_posts.csv to view posting trends over time.")
        return
    if posts.empty or "posted_at" not in posts.columns:
        st.info("No posts available for time-series visualization.")
        return
    posts = posts.copy()
    posts["month"] = pd.to_datetime(posts["posted_at"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    per_month = posts.groupby("month").size().reset_index(name="posts")
    chart = alt.Chart(per_month).mark_line(point=True).encode(x="month", y="posts")
    st.altair_chart(chart, use_container_width=True)


def main():
    st.set_page_config(page_title="Newborn Product Research", layout="wide")
    st.title("Newborn/Baby Product Market Research Dashboard")
    st.caption("Data sourced from local CSV exports; no live scraping performed.")

    df = load_data()
    filters = sidebar_filters(df)
    filtered_df = apply_filters(df, filters)

    render_kpis(filtered_df, posts_total=int(df["num_posts"].sum()))

    st.markdown("---")
    render_top_products_table(filtered_df)

    st.markdown("---")
    st.subheader("Category and brand trends")
    c1, c2 = st.columns(2)
    with c1:
        chart_top_categories(filtered_df)
    with c2:
        chart_top_brands(filtered_df)

    st.markdown("---")
    posts = load_posts_for_time_series()
    chart_time_series(posts)


if __name__ == "__main__":
    main()
