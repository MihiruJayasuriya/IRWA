"""
preprocess.py
Reads ALL raw CSVs from ../data/raw/ (consumption, rainfall, reservoir —
Person A's Collector Agent mock data), merges them into one wide table
keyed on (zone, date), cleans it, engineers time-series features, splits
into train/test by date, and writes the results to ../data/processed/.

Run this file directly:
    python preprocess.py
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to this file, so it works regardless of where you run it from)
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(THIS_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(THIS_DIR, "..", "data", "processed")

# ---------------------------------------------------------------------------
# Flexible column detection — Kaggle/mock CSVs vary in exact header names.
# Edit these lists if a real CSV's column names aren't matched.
# ---------------------------------------------------------------------------
ZONE_PATTERNS = ["zone", "area", "region", "district", "location"]
DATE_PATTERNS = ["date", "timestamp", "time", "day"]
CONSUMPTION_PATTERNS = ["consumption", "usage", "liters", "litres", "demand", "flow"]
RAINFALL_PATTERNS = ["rainfall", "precip", "rain_mm", "rain"]
# NOTE: order matters here. The new reservoir CSV added a "reservoir_name"
# column (e.g. "Flathead Lake") alongside "reservoir_pct_full". Both contain
# the word "reservoir", so the specific patterns must be checked BEFORE the
# generic "reservoir" pattern, or "reservoir_name" (text) gets picked instead
# of "reservoir_pct_full" (the number we actually need).
RESERVOIR_PATTERNS = ["reservoir_pct", "pct_full", "reservoir_level", "tank_level", "reservoir"]


def _find_column(columns, patterns):
    lower_map = {c.lower(): c for c in columns}
    for pattern in patterns:
        for lower_name, original_name in lower_map.items():
            if pattern in lower_name:
                return original_name
    return None


def _find_csv(patterns):
    """Finds the first CSV in data/raw/ whose filename matches one of the given patterns."""
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"Folder not found: {RAW_DIR}")
    for fname in os.listdir(RAW_DIR):
        if not fname.lower().endswith(".csv"):
            continue
        for pattern in patterns:
            if pattern in fname.lower():
                return os.path.join(RAW_DIR, fname)
    return None


def load_consumption() -> pd.DataFrame:
    path = _find_csv(["consumption", "usage", "water_consumption"])
    if path is None:
        raise FileNotFoundError(
            f"No consumption CSV found in {RAW_DIR}. This file is required — "
            f"it's what we're forecasting."
        )
    raw = pd.read_csv(path)
    zone_col = _find_column(raw.columns, ZONE_PATTERNS)
    date_col = _find_column(raw.columns, DATE_PATTERNS)
    value_col = _find_column(raw.columns, CONSUMPTION_PATTERNS)
    missing = [n for n, c in [("zone", zone_col), ("date", date_col), ("consumption", value_col)] if c is None]
    if missing:
        raise ValueError(f"consumption CSV ({path}): could not detect column(s) {missing}. "
                          f"Actual columns: {list(raw.columns)}")
    df = pd.DataFrame({
        "zone": raw[zone_col].astype(str).str.strip(),
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        "consumption": pd.to_numeric(raw[value_col], errors="coerce"),
    })
    print(f"Loaded consumption: {path} — shape {df.shape}")
    return df


def load_optional_signal(filename_patterns, column_patterns, output_name):
    """
    Loads an optional extra signal (rainfall, reservoir level). If the file
    isn't present in data/raw/, returns None and preprocessing continues
    without it — this file is a nice-to-have, not a hard requirement.
    """
    path = _find_csv(filename_patterns)
    if path is None:
        print(f"[INFO] No {output_name} CSV found in {RAW_DIR} — continuing without it.")
        return None

    raw = pd.read_csv(path)
    zone_col = _find_column(raw.columns, ZONE_PATTERNS)
    date_col = _find_column(raw.columns, DATE_PATTERNS)
    value_col = _find_column(raw.columns, column_patterns)
    missing = [n for n, c in [("zone", zone_col), ("date", date_col), (output_name, value_col)] if c is None]
    if missing:
        print(f"[WARN] {output_name} CSV ({path}): could not detect column(s) {missing}. "
              f"Actual columns: {list(raw.columns)}. Skipping this signal.")
        return None

    df = pd.DataFrame({
        "zone": raw[zone_col].astype(str).str.strip(),
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        output_name: pd.to_numeric(raw[value_col], errors="coerce"),
    })
    print(f"Loaded {output_name}: {path} — shape {df.shape}")
    return df


def merge_all() -> pd.DataFrame:
    consumption = load_consumption()
    rainfall = load_optional_signal(["rainfall", "rain"], RAINFALL_PATTERNS, "rainfall_mm")
    reservoir = load_optional_signal(["reservoir", "tank"], RESERVOIR_PATTERNS, "reservoir_pct_full")

    df = consumption
    for extra in (rainfall, reservoir):
        if extra is not None:
            df = df.merge(extra, on=["zone", "date"], how="left")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["zone", "date", "consumption"])
    df = df.drop_duplicates(subset=["zone", "date"])
    df = df[df["consumption"] >= 0]
    after = len(df)
    print(f"Dropped {before - after} rows (missing core values, duplicates, or negative consumption)")

    # rainfall/reservoir can be missing for some rows even after the merge
    # (e.g. a sensor outage) — fill with each zone's own median rather than
    # dropping rows, since consumption is still valid even if a side signal
    # is missing for that day
    for col in ("rainfall_mm", "reservoir_pct_full"):
        if col in df.columns and df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            df[col] = df.groupby("zone")[col].transform(lambda s: s.fillna(s.median()))
            print(f"Filled {missing_count} missing '{col}' values with each zone's median")

    df = df.sort_values(["zone", "date"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds time-series + optional environmental features."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["consumption_lag_1"] = df.groupby("zone")["consumption"].shift(1)
    df["consumption_rolling_7"] = (
        df.groupby("zone")["consumption"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )

    if "rainfall_mm" in df.columns:
        df["rainfall_rolling_7"] = (
            df.groupby("zone")["rainfall_mm"]
            .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        )

    # drop the first row per zone where lag_1 is NaN (no prior reading yet)
    df = df.dropna(subset=["consumption_lag_1"]).reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Splits by date, not randomly — the last `test_fraction` of each
    zone's timeline becomes the test set, avoiding leakage of future
    values into training.
    """
    train_parts, test_parts = [], []
    for zone, zone_df in df.groupby("zone"):
        zone_df = zone_df.sort_values("date")
        split_idx = int(len(zone_df) * (1 - test_fraction))
        train_parts.append(zone_df.iloc[:split_idx])
        test_parts.append(zone_df.iloc[split_idx:])
    train_df = pd.concat(train_parts).sort_values(["zone", "date"]).reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values(["zone", "date"]).reset_index(drop=True)
    return train_df, test_df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = merge_all()
    df = clean(df)
    df = engineer_features(df)

    train_df, test_df = time_based_split(df, test_fraction=0.2)

    clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    df.to_csv(clean_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nFinal columns: {list(df.columns)}")
    print(f"\nSaved:")
    print(f"  {clean_path}  ({len(df)} rows)")
    print(f"  {train_path}  ({len(train_df)} rows)")
    print(f"  {test_path}   ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
