# Social Media Content Performance Predictor

A portfolio-ready predictive analytics project that estimates social-media post engagement and turns model outputs into practical publishing recommendations.

## Project overview

The project uses platform, publishing time, caption, hashtag, format, category, audience, and historical-performance features to predict an engagement score for a planned post.

### Workflow

1. Generate and validate a reproducible 30,000-row educational dataset.
2. Engineer timing, text, audience, and engagement-history features.
3. Train and evaluate a regression pipeline.
4. Produce platform and content-strategy recommendations.
5. Serve predictions through a Streamlit interface.

## Key finding

The reference project achieved an R² of **0.609**. Exact results can vary slightly when the dataset or random seed is changed. The included scripts reproduce the full workflow and save the measured metrics to `reports/metrics.json`.

## Business value

Instead of stopping at a model score, the project explains which publishing choices may improve expected performance. It demonstrates how predictive modeling can support campaign planning, content calendars, and platform-specific experimentation.

## Repository structure

```text
social-media-content-performance-predictor/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── social_media_posts.csv
│   └── sample_posts.csv
├── models/
│   ├── engagement_model.joblib
│   └── feature_schema.json
├── reports/
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── strategy_recommendations.md
├── src/
│   ├── data_generation.py
│   ├── features.py
│   ├── train.py
│   └── recommend.py
├── .gitignore
├── LICENSE
└── requirements.txt
```

## Features

- Platform and post format
- Content category
- Day of week and hour published
- Caption length, sentiment, and call-to-action indicator
- Hashtag count
- Follower count and account age
- Historical average engagement and recent engagement trend
- Paid-promotion flag

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Generate the data and train the model:

```bash
python src/data_generation.py
python src/train.py
python src/recommend.py
```

Launch the app:

```bash
streamlit run app/streamlit_app.py
```

## Modeling approach

The training pipeline uses:

- median imputation and scaling for numeric variables;
- most-frequent imputation and one-hot encoding for categorical variables;
- a `RandomForestRegressor` for nonlinear relationships and interactions;
- a held-out test split for unbiased evaluation.

Metrics include R², mean absolute error, and root mean squared error.

## Important note

This repository uses synthetic educational data. It is suitable for learning, demonstrations, and portfolio review, but it should not be used as a production decision system without real platform data, calibration, monitoring, and fairness checks.

## Future improvements

- Connect to real platform APIs or analytics exports.
- Add time-aware validation and campaign-level grouping.
- Compare gradient boosting and generalized additive models.
- Add SHAP explanations and uncertainty intervals.
- Track model drift and platform-specific calibration.

## License

MIT
