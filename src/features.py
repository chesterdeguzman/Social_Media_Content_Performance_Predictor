"""Shared feature definitions and validation helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

TARGET = "engagement_score"

NUMERIC_FEATURES = [
    "publish_hour",
    "caption_length",
    "hashtag_count",
    "caption_sentiment",
    "follower_count",
    "account_age_months",
    "historical_avg_engagement",
    "recent_engagement_trend",
    "paid_promotion",
]

CATEGORICAL_FEATURES = [
    "platform",
    "day_of_week",
    "post_format",
    "content_category",
    "has_call_to_action",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ALLOWED_VALUES = {
    "platform": ["Instagram", "TikTok", "Facebook", "LinkedIn", "X"],
    "day_of_week": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    "post_format": ["Image", "Carousel", "Short Video", "Long Video", "Text"],
    "content_category": [
        "Education",
        "Entertainment",
        "Lifestyle",
        "Product",
        "Community",
        "Behind the Scenes",
    ],
    "has_call_to_action": ["Yes", "No"],
}


def validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise a useful error when required columns are missing."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select features in a stable order and normalize simple types."""
    validate_columns(frame, FEATURES)
    result = frame.loc[:, FEATURES].copy()
    result["paid_promotion"] = pd.to_numeric(result["paid_promotion"], errors="coerce")
    return result
