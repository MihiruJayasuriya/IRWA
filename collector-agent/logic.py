"""
logic.py - Collector Agent core logic (robustness-patched version)

Changes from the original:
  - Absolute paths (os.path.dirname(__file__)) so this works regardless
    of which directory uvicorn is launched from.
  - SQLite connections opened via context managers (`with`), so they
    always close properly even if an error occurs mid-query.
  - Explicit FileNotFoundError with a clear message if a CSV is missing,
    instead of a confusing pandas traceback.
"""

import itertools
import os
import sqlite3
from datetime import datetime, timezone
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORICAL_DB_PATH = os.path.join(BASE_DIR, "historical_usage.db")
HISTORICAL_SPLIT_RATIO = 0.8


def load_merged_data():
    consumption_path = os.path.join(DATA_DIR, "water_consumption_forecasting.csv")
    rainfall_path = os.path.join(DATA_DIR, "rainfall.csv")
    reservoir_path = os.path.join(DATA_DIR, "reservoir_levels.csv")

    for p in [consumption_path, rainfall_path, reservoir_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required dataset missing: {p}")

    consumption = pd.read_csv(consumption_path)
    rainfall = pd.read_csv(rainfall_path)
    reservoir = pd.read_csv(reservoir_path)

    merged = consumption.merge(rainfall, on=["date", "region"], how="left")
    merged = merged.merge(reservoir, on=["date", "region"], how="left")
    merged = merged.sort_values(["date", "region"]).reset_index(drop=True)
    return merged


def split_and_store(merged_df):
    unique_dates = sorted(merged_df["date"].unique())
    split_index = int(len(unique_dates) * HISTORICAL_SPLIT_RATIO)
    historical_dates = set(unique_dates[:split_index])
    live_dates = unique_dates[split_index:]

    historical_df = merged_df[merged_df["date"].isin(historical_dates)].copy()
    live_df = merged_df[merged_df["date"].isin(live_dates)].copy()

    with sqlite3.connect(HISTORICAL_DB_PATH) as conn:
        historical_df.to_sql("readings", conn, if_exists="replace", index=False)

    return historical_df, live_df


merged_df = load_merged_data()
historical_df, live_df = split_and_store(merged_df)
row_iterator = itertools.cycle(live_df.to_dict(orient="records"))


def get_next_reading():
    row = next(row_iterator)
    return {
        "agent": "collector",
        "zone": row["region"],
        "type": "usage_reading",
        "payload": {
            "date": row["date"],
            "consumption_liters": float(row["consumption_liters"]),
            "rainfall_mm": float(row["rainfall_mm"]),
            "reservoir_pct_full": float(row["reservoir_pct_full"]),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_historical_baseline(region: str):
    clean_region = str(region).strip()

    with sqlite3.connect(HISTORICAL_DB_PATH) as conn:
        cursor = conn.cursor()
        query = """
            SELECT
                region,
                AVG(consumption_liters) AS avg_consumption_liters,
                AVG(rainfall_mm) AS avg_rainfall_mm,
                AVG(reservoir_pct_full) AS avg_reservoir_pct_full,
                COUNT(*) AS num_historical_records
            FROM readings
            WHERE LOWER(region) = LOWER(?)
            GROUP BY region
        """
        result = cursor.execute(query, (clean_region,)).fetchone()

    if result is None:
        return None

    return {
        "region": result[0],
        "avg_consumption_liters": round(result[1], 2),
        "avg_rainfall_mm": round(result[2], 2),
        "avg_reservoir_pct_full": round(result[3], 2),
        "num_historical_records": result[4],
    }


def get_summary():
    return {
        "total_rows": len(merged_df),
        "historical_rows": len(historical_df),
        "live_rows": len(live_df),
        "regions": sorted(merged_df["region"].unique().tolist()),
        "date_range": [str(merged_df["date"].min()), str(merged_df["date"].max())],
        "historical_date_range": [
            str(historical_df["date"].min()),
            str(historical_df["date"].max()),
        ],
        "live_date_range": [str(live_df["date"].min()), str(live_df["date"].max())],
    }
