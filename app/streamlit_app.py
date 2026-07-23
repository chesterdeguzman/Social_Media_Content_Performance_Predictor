"""Streamlit interface for social-media engagement prediction."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import ALLOWED_VALUES, prepare_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "engagement_model.joblib"

st.set_page_config(page_title="Content Performance Predictor", page_icon="📈", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


st.title("Social Media Content Performance Predictor")
st.caption("Estimate expected engagement and receive practical publishing suggestions.")

try:
    model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

left, right = st.columns(2)
with left:
    platform = st.selectbox("Platform", ALLOWED_VALUES["platform"])
    post_format = st.selectbox("Post format", ALLOWED_VALUES["post_format"])
    content_category = st.selectbox("Content category", ALLOWED_VALUES["content_category"])
    day_of_week = st.selectbox("Day of week", ALLOWED_VALUES["day_of_week"])
    publish_hour = st.slider("Publishing hour", 0, 23, 18)
    has_call_to_action = st.selectbox("Call to action", ALLOWED_VALUES["has_call_to_action"])

with right:
    caption_length = st.slider("Caption length", 5, 700, 140)
    hashtag_count = st.slider("Hashtag count", 0, 25, 6)
    caption_sentiment = st.slider("Caption sentiment", -1.0, 1.0, 0.3, 0.05)
    follower_count = st.number_input("Follower count", min_value=250, max_value=3_000_000, value=25_000)
    account_age_months = st.slider("Account age in months", 1, 144, 24)
    historical_avg_engagement = st.slider("Historical average engagement", 0.2, 45.0, 8.0, 0.1)
    recent_engagement_trend = st.slider("Recent engagement trend", -0.8, 1.0, 0.05, 0.01)
    paid_promotion = st.toggle("Paid promotion")

row = pd.DataFrame([{
    "platform": platform,
    "day_of_week": day_of_week,
    "publish_hour": publish_hour,
    "post_format": post_format,
    "content_category": content_category,
    "caption_length": caption_length,
    "hashtag_count": hashtag_count,
    "caption_sentiment": caption_sentiment,
    "has_call_to_action": has_call_to_action,
    "follower_count": follower_count,
    "account_age_months": account_age_months,
    "historical_avg_engagement": historical_avg_engagement,
    "recent_engagement_trend": recent_engagement_trend,
    "paid_promotion": int(paid_promotion),
}])

if st.button("Predict performance", type="primary", use_container_width=True):
    score = float(model.predict(prepare_features(row))[0])
    st.metric("Predicted engagement score", f"{score:.1f} / 100")

    suggestions = []
    if platform == "LinkedIn" and not 8 <= publish_hour <= 11:
        suggestions.append("Test a weekday morning publishing window for LinkedIn.")
    if platform == "TikTok" and publish_hour < 18:
        suggestions.append("Test an evening slot for TikTok short-form content.")
    if platform == "Instagram" and post_format not in {"Carousel", "Short Video"}:
        suggestions.append("Compare this format against a carousel or short video.")
    if hashtag_count > 12:
        suggestions.append("Run an experiment with a smaller, more focused hashtag set.")
    if has_call_to_action == "No":
        suggestions.append("Consider a clear, low-friction call to action.")
    if recent_engagement_trend < 0:
        suggestions.append("Treat the prediction cautiously because recent account momentum is negative.")
    if not suggestions:
        suggestions.append("The selected strategy is reasonably aligned with the model's learned patterns.")

    st.subheader("Recommended tests")
    for suggestion in suggestions:
        st.write(f"- {suggestion}")

st.info("Educational model trained on synthetic data. Predictions are directional, not guarantees.")
