"""
Utility functions for text cleaning and lightweight similarity scoring.

These helpers are intentionally lightweight so they can run in constrained
execution environments without requiring heavyweight NLP dependencies.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, List, Optional

try:  # Optional dependency; falls back to difflib if unavailable.
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional path
    fuzz = None  # type: ignore


def clean_text(text: Optional[str]) -> str:
    """Normalize text for downstream matching.

    Steps:
    - Convert to lowercase.
    - Strip accents.
    - Replace non-alphanumeric characters with spaces.
    - Collapse multiple spaces.
    - Trim leading/trailing whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split cleaned text into tokens."""
    return [tok for tok in clean_text(text).split(" ") if tok]


def contains_keywords(text: str, keywords: Iterable[str]) -> bool:
    """Check if any keyword is present as a full token in the text."""
    text_tokens = set(tokenize(text))
    keywords_clean = {clean_text(k) for k in keywords if k}
    return any(k in text_tokens for k in keywords_clean)


def similarity(a: str, b: str) -> float:
    """Return a 0-100 similarity score between two strings.

    Uses rapidfuzz if available for speed; otherwise falls back to
    difflib.SequenceMatcher ratio scaled to 0-100.
    """
    a_clean, b_clean = clean_text(a), clean_text(b)
    if not a_clean or not b_clean:
        return 0.0
    if fuzz:
        return float(fuzz.token_set_ratio(a_clean, b_clean))
    ratio = SequenceMatcher(None, a_clean, b_clean).ratio()
    return ratio * 100


def best_match(candidate: str, corpus: Iterable[str]) -> Optional[float]:
    """Return the best similarity score for a candidate against a corpus.

    Args:
        candidate: Text to compare.
        corpus: Iterable of comparison strings.

    Returns:
        Highest similarity score or None if corpus empty.
    """
    scores: List[float] = []
    for item in corpus:
        score = similarity(candidate, item)
        scores.append(score)
    if not scores:
        return None
    return max(scores)
