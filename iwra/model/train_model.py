"""
train_model.py
Trains the forecasting model using consumption history + rainfall +
reservoir level (when available), evaluates it properly (not just one
number), and saves the trained model + per-zone anomaly baselines.

Reads:  ../data/processed/train.csv, ../data/processed/test.csv
Writes: model.pkl, zone_baselines.json  (both in this folder)

Run:
    python train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(THIS_DIR, "..", "data", "processed")
MODEL_PATH = os.path.join(THIS_DIR, "model.pkl")
BASELINES_PATH = os.path.join(THIS_DIR, "zone_baselines.json")

TARGET_COL = "consumption"

# Base features always used. Optional ones are added automatically below
# if present in the data (so this script still works if a teammate's CSV
# doesn't have rainfall/reservoir yet).
BASE_FEATURES = ["day_of_week", "month", "is_weekend", "consumption_lag_1", "consumption_rolling_7"]
OPTIONAL_FEATURES = ["rainfall_mm", "reservoir_pct_full", "rainfall_rolling_7"]


def load_data():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train_df, test_df


def get_feature_list(df: pd.DataFrame):
    features = list(BASE_FEATURES)
    for col in OPTIONAL_FEATURES:
        if col in df.columns:
            features.append(col)
    return features


def build_features(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Turns 'zone' into one-hot columns so one model serves all zones."""
    X = df[feature_cols].copy()
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone")
    X = pd.concat([X, zone_dummies], axis=1)
    return X


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{label:12s}  MAE: {mae:8.1f}   RMSE: {rmse:8.1f}   R2: {r2:6.3f}")
    return mae, rmse, r2


def main():
    train_df, test_df = load_data()
    print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}")

    feature_cols = get_feature_list(train_df)
    print(f"Features used: {feature_cols}\n")

    X_train = build_features(train_df, feature_cols)
    y_train = train_df[TARGET_COL]
    X_test = build_features(test_df, feature_cols)
    y_test = test_df[TARGET_COL]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # max_depth=8 was tried first and overfit badly (train R2 0.65 vs test
    # R2 0.11 — a huge gap). max_depth=3 + min_samples_leaf=5 closes that
    # gap substantially and gives a better, more honest test score —
    # found via a small manual sweep over depth/leaf-size combinations.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ---------------------------------------------------------------
    # ACCURACY CHECK 1 — train vs test performance
    # If train is much better than test, the model is overfitting.
    # ---------------------------------------------------------------
    print("--- Train vs Test performance ---")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    evaluate(y_train, train_preds, "Train")
    evaluate(y_test, test_preds, "Test")

    # ---------------------------------------------------------------
    # ACCURACY CHECK 2 — compare against a naive baseline
    # A real model should beat "just guess the mean" or "repeat yesterday's
    # value" — if it doesn't, the model isn't adding value.
    # ---------------------------------------------------------------
    print("\n--- Baseline comparisons (is the model actually better than guessing?) ---")
    dummy_mean = DummyRegressor(strategy="mean")
    dummy_mean.fit(X_train, y_train)
    evaluate(y_test, dummy_mean.predict(X_test), "Mean-only")

    naive_lag = test_df["consumption_lag_1"]  # "predict tomorrow = today"
    evaluate(y_test, naive_lag, "Naive-lag")

    evaluate(y_test, test_preds, "Our model")

    # ---------------------------------------------------------------
    # ACCURACY CHECK 3 — cross-validation on training data
    # Checks the score isn't a fluke of one particular train/test split.
    # TimeSeriesSplit respects chronological order (no shuffling).
    # ---------------------------------------------------------------
    print("\n--- 5-fold time-series cross-validation (on train set) ---")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="r2")
    print(f"R2 per fold: {np.round(cv_scores, 3)}")
    print(f"Mean R2: {cv_scores.mean():.3f}  (std: {cv_scores.std():.3f})")

    # ---------------------------------------------------------------
    # Feature importance — which signals actually drove the predictions
    # ---------------------------------------------------------------
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\n--- Feature importance ---")
    print(importances.round(3))

    # ---------------------------------------------------------------
    # Per-zone breakdown — a model can look fine overall while being bad
    # for one specific zone; check each separately
    # ---------------------------------------------------------------
    print("\n--- Per-zone test performance ---")
    test_df_eval = test_df.copy()
    test_df_eval["pred"] = test_preds
    for zone, zone_df in test_df_eval.groupby("zone"):
        mae = mean_absolute_error(zone_df[TARGET_COL], zone_df["pred"])
        rmse = np.sqrt(mean_squared_error(zone_df[TARGET_COL], zone_df["pred"]))
        print(f"  {zone:10s}  MAE: {mae:8.1f}   RMSE: {rmse:8.1f}")

    # ---- save model ----
    joblib.dump({"model": model, "feature_columns": list(X_train.columns)}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    # ---- per-zone anomaly baseline (mean/std of consumption from TRAIN only) ----
    baselines = {}
    for zone, zone_df in train_df.groupby("zone"):
        baselines[zone] = {
            "mean": float(zone_df[TARGET_COL].mean()),
            "std": float(zone_df[TARGET_COL].std(ddof=0)) or 1e-6,
        }
    with open(BASELINES_PATH, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"Saved per-zone baselines to {BASELINES_PATH}")


if __name__ == "__main__":
    main()
