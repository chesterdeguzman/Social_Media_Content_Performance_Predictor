"""Generate a reproducible synthetic social-media performance dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "social_media_posts.csv"
SAMPLE_PATH = ROOT / "data" / "sample_posts.csv"

PLATFORMS = ["Instagram", "TikTok", "Facebook", "LinkedIn", "X"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FORMATS = ["Image", "Carousel", "Short Video", "Long Video", "Text"]
CATEGORIES = ["Education", "Entertainment", "Lifestyle", "Product", "Community", "Behind the Scenes"]


def generate_dataset(rows: int = 30_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    platform = rng.choice(PLATFORMS, rows, p=[0.29, 0.22, 0.18, 0.16, 0.15])
    day = rng.choice(DAYS, rows)
    post_format = rng.choice(FORMATS, rows, p=[0.27, 0.18, 0.29, 0.09, 0.17])
    category = rng.choice(CATEGORIES, rows)
    publish_hour = rng.integers(0, 24, rows)
    caption_length = np.clip(rng.gamma(2.2, 65, rows), 5, 700).round().astype(int)
    hashtag_count = np.clip(rng.poisson(5, rows), 0, 25)
    caption_sentiment = np.clip(rng.normal(0.25, 0.38, rows), -1, 1)
    follower_count = np.clip(np.exp(rng.normal(9.15, 1.25, rows)), 250, 3_000_000).round().astype(int)
    account_age_months = rng.integers(1, 145, rows)
    historical_avg = np.clip(rng.beta(2.2, 8.2, rows) * 100, 0.2, 45)
    recent_trend = np.clip(rng.normal(0, 0.24, rows), -0.8, 1.0)
    paid = rng.binomial(1, 0.11, rows)
    cta = rng.choice(["Yes", "No"], rows, p=[0.58, 0.42])

    platform_effect = pd.Series(platform).map({
        "Instagram": 4.2, "TikTok": 7.5, "Facebook": 1.8, "LinkedIn": 3.3, "X": 2.5
    }).to_numpy()
    format_effect = pd.Series(post_format).map({
        "Image": 1.8, "Carousel": 4.6, "Short Video": 7.1, "Long Video": 2.2, "Text": 0.8
    }).to_numpy()
    category_effect = pd.Series(category).map({
        "Education": 3.4, "Entertainment": 5.2, "Lifestyle": 3.8,
        "Product": 1.6, "Community": 4.0, "Behind the Scenes": 4.5
    }).to_numpy()
    day_effect = pd.Series(day).map({
        "Monday": 0.4, "Tuesday": 1.0, "Wednesday": 1.4, "Thursday": 1.5,
        "Friday": 1.0, "Saturday": 1.8, "Sunday": 1.2
    }).to_numpy()

    evening_bonus = np.where((publish_hour >= 18) & (publish_hour <= 21), 3.0, 0.0)
    lunch_bonus = np.where((publish_hour >= 11) & (publish_hour <= 13), 1.5, 0.0)
    platform_time_interaction = np.where(
        ((platform == "LinkedIn") & (publish_hour >= 8) & (publish_hour <= 11)) |
        ((platform == "TikTok") & (publish_hour >= 19) & (publish_hour <= 23)),
        2.8,
        0.0,
    )
    format_platform_interaction = np.where(
        ((platform == "Instagram") & (post_format == "Carousel")) |
        ((platform == "TikTok") & (post_format == "Short Video")) |
        ((platform == "LinkedIn") & (post_format == "Text")),
        3.2,
        0.0,
    )
    caption_fit = -0.00008 * np.square(caption_length - np.where(platform == "LinkedIn", 230, 115))
    hashtag_fit = -0.09 * np.square(hashtag_count - np.where(platform == "Instagram", 8, 4))
    follower_scale = 1.0 + 0.8 * np.log10(follower_count) / 6

    signal = (
        7.0
        + platform_effect
        + format_effect
        + category_effect
        + day_effect
        + evening_bonus
        + lunch_bonus
        + platform_time_interaction
        + format_platform_interaction
        + caption_fit
        + hashtag_fit
        + 2.2 * caption_sentiment
        + 1.4 * (cta == "Yes")
        + 4.5 * paid
        + 0.44 * historical_avg
        + 9.0 * recent_trend
        + 0.018 * np.sqrt(account_age_months)
    ) * follower_scale

    noise = rng.normal(0, 7.8, rows)
    engagement_score = np.clip(signal + noise, 0, 100)

    frame = pd.DataFrame({
        "platform": platform,
        "day_of_week": day,
        "publish_hour": publish_hour,
        "post_format": post_format,
        "content_category": category,
        "caption_length": caption_length,
        "hashtag_count": hashtag_count,
        "caption_sentiment": caption_sentiment.round(3),
        "has_call_to_action": cta,
        "follower_count": follower_count,
        "account_age_months": account_age_months,
        "historical_avg_engagement": historical_avg.round(3),
        "recent_engagement_trend": recent_trend.round(3),
        "paid_promotion": paid,
        "engagement_score": engagement_score.round(3),
    })
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    if len(frame) != 30_000:
        raise ValueError(f"Expected 30,000 rows, found {len(frame):,}.")
    if frame.isna().any().any():
        raise ValueError("Generated dataset contains missing values.")
    if not frame["engagement_score"].between(0, 100).all():
        raise ValueError("Engagement score must be between 0 and 100.")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset()
    validate_dataset(frame)
    frame.to_csv(OUTPUT_PATH, index=False)
    frame.sample(20, random_state=42).to_csv(SAMPLE_PATH, index=False)
    print(f"Saved {len(frame):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
