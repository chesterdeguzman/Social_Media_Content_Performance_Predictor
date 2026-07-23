"""Generate simple strategy recommendations from the synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "social_media_posts.csv"
OUTPUT_PATH = ROOT / "reports" / "strategy_recommendations.md"


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    lines = [
        "# Platform and Content Strategy Recommendations",
        "",
        "These recommendations are descriptive summaries of the synthetic educational dataset.",
        "They should be validated through real-world experiments before use.",
        "",
    ]

    for platform, group in frame.groupby("platform"):
        best_hours = group.groupby("publish_hour")["engagement_score"].mean().nlargest(3)
        best_formats = group.groupby("post_format")["engagement_score"].mean().nlargest(2)
        best_categories = group.groupby("content_category")["engagement_score"].mean().nlargest(2)
        lines.extend([
            f"## {platform}",
            "",
            f"- Strongest hours: {', '.join(str(int(hour)) + ':00' for hour in best_hours.index)}",
            f"- Strongest formats: {', '.join(best_formats.index)}",
            f"- Strongest categories: {', '.join(best_categories.index)}",
            "",
        ])

    history_corr = frame[["historical_avg_engagement", "engagement_score"]].corr().iloc[0, 1]
    lines.extend([
        "## Cross-platform observation",
        "",
        f"Historical average engagement has a correlation of **{history_corr:.3f}** with the target in this dataset,",
        "supporting its use as an important contextual feature rather than a universal causal driver.",
        "",
    ])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved recommendations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
