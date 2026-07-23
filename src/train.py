"""Train and evaluate the engagement prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    ALLOWED_VALUES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    prepare_features,
    validate_columns,
)

DATA_PATH = ROOT / "data" / "social_media_posts.csv"
MODEL_PATH = ROOT / "models" / "engagement_model.joblib"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=16,
        min_samples_leaf=4,
        max_features=0.75,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def save_feature_importance(pipeline: Pipeline) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(IMPORTANCE_PATH, index=False)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run `python src/data_generation.py` first.")

    frame = pd.read_csv(DATA_PATH)
    validate_columns(frame, FEATURES + [TARGET])
    X = prepare_features(frame)
    y = frame[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "rows": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4),
        "random_state": 42,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    SCHEMA_PATH.write_text(
        json.dumps({"features": FEATURES, "allowed_values": ALLOWED_VALUES}, indent=2),
        encoding="utf-8",
    )
    save_feature_importance(pipeline)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
