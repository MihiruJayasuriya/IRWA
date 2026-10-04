"""
Analysis Agent - core logic (v3)
Loads the TRAINED model (model.pkl) and per-zone baselines
(zone_baselines.json). The forecast now also uses rainfall and
reservoir level, since the retrained model includes those features.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import deque

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
BASELINES_PATH = os.path.join(MODEL_DIR, "zone_baselines.json")


class AnalysisEngine:
    def __init__(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(BASELINES_PATH):
            raise FileNotFoundError(
                f"Model files not found in {MODEL_DIR} — run "
                f"`python train_model.py` inside the model/ folder first."
            )

        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        # The trained model was saved with n_jobs=-1. Inference needs no
        # process pool and some Windows environments deny worker creation.
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.feature_columns = bundle["feature_columns"]
        self.uses_environmental = any(
            c in self.feature_columns for c in ("rainfall_mm", "reservoir_pct_full", "rainfall_rolling_7")
        )

        with open(BASELINES_PATH) as f:
            self.baselines = json.load(f)

        self.zones = list(self.baselines.keys())

        # rolling recent-reading history per zone (consumption + rainfall),
        # seeded from the baseline mean so early requests still work
        self.recent_history = {
            zone: deque([stats["mean"]] * 7, maxlen=30)
            for zone, stats in self.baselines.items()
        }
        # default environmental values used only if the caller doesn't supply
        # rainfall/reservoir for a forecast — sensible neutral placeholders
        self.recent_rainfall = {zone: deque([0.0] * 7, maxlen=7) for zone in self.zones}
        self.recent_reservoir = {zone: 70.0 for zone in self.zones}

    # -------- job 1: anomaly detection --------
    def score_reading(self, zone: str, value: float) -> dict:
        if zone not in self.baselines:
            return {
                "zone": zone,
                "value": value,
                "z_score": None,
                "is_anomaly": False,
                "severity": "unknown",
                "note": f"zone '{zone}' not in trained baselines: {self.zones}",
            }

        mean = self.baselines[zone]["mean"]
        std = self.baselines[zone]["std"] or 1e-6
        z = (value - mean) / std
        abs_z = abs(z)

        if abs_z >= 4:
            severity = "critical"
        elif abs_z >= 3:
            severity = "high"
        elif abs_z >= 2.5:
            severity = "moderate"
        else:
            severity = "normal"

        self.recent_history[zone].append(value)

        return {
            "zone": zone,
            "value": value,
            "z_score": round(z, 2),
            "is_anomaly": abs_z >= 2.5,
            "severity": severity,
            "baseline_mean": round(mean, 1),
            "baseline_std": round(std, 1),
        }

    # -------- job 2: forecasting (uses the TRAINED model) --------
    def forecast_zone(self, zone: str, days_ahead: int = 7,
                       rainfall_mm: float = None, reservoir_pct_full: float = None) -> dict:
        """
        rainfall_mm / reservoir_pct_full are optional. If the caller
        (Router/Collector) supplies current readings, they're used as the
        forecast's starting environmental context. If not, we fall back to
        each zone's recent-average rainfall and a neutral reservoir level.
        """
        if zone not in self.baselines:
            return {"zone": zone, "error": f"zone '{zone}' not in trained baselines: {self.zones}"}

        history = list(self.recent_history[zone])

        if rainfall_mm is not None:
            self.recent_rainfall[zone].append(rainfall_mm)
        if reservoir_pct_full is not None:
            self.recent_reservoir[zone] = reservoir_pct_full

        rainfall_now = rainfall_mm if rainfall_mm is not None else float(np.mean(self.recent_rainfall[zone]))
        reservoir_now = self.recent_reservoir[zone]

        forecasts = []
        for day_offset in range(1, days_ahead + 1):
            lag_1 = history[-1]
            rolling_7 = float(np.mean(history[-7:]))

            row = {col: 0 for col in self.feature_columns}
            row["day_of_week"] = day_offset % 7
            row["month"] = 1  # placeholder — refine if you track the actual forecast date
            row["is_weekend"] = 1 if (day_offset % 7) in (5, 6) else 0
            row["consumption_lag_1"] = lag_1
            row["consumption_rolling_7"] = rolling_7
            if "rainfall_mm" in row:
                row["rainfall_mm"] = rainfall_now
            if "rainfall_rolling_7" in row:
                row["rainfall_rolling_7"] = float(np.mean(self.recent_rainfall[zone])) if self.recent_rainfall[zone] else rainfall_now
            if "reservoir_pct_full" in row:
                row["reservoir_pct_full"] = reservoir_now
            zone_col = f"zone_{zone}"
            if zone_col in row:
                row[zone_col] = 1

            X = pd.DataFrame([[row[col] for col in self.feature_columns]], columns=self.feature_columns)
            pred = float(self.model.predict(X)[0])
            pred = max(0.0, pred)
            forecasts.append(round(pred, 1))
            history.append(pred)

        mean = self.baselines[zone]["mean"]
        std = self.baselines[zone]["std"]
        historical_p90 = mean + 1.28 * std
        peak_forecast = max(forecasts)

        if peak_forecast >= historical_p90 * 1.15:
            risk = "high"
        elif peak_forecast >= historical_p90:
            risk = "moderate"
        else:
            risk = "low"

        # low reservoir + rising demand is a stronger shortage signal than
        # demand alone — bump risk up a tier if the reservoir is already low
        if reservoir_now < 40 and risk == "moderate":
            risk = "high"
        elif reservoir_now < 40 and risk == "low":
            risk = "moderate"

        avg_first_half = np.mean(forecasts[:len(forecasts) // 2 or 1])
        avg_second_half = np.mean(forecasts[len(forecasts) // 2:])
        trend_direction = "rising" if avg_second_half > avg_first_half else (
            "falling" if avg_second_half < avg_first_half else "flat"
        )

        return {
            "zone": zone,
            "days_ahead": days_ahead,
            "forecast": forecasts,
            "trend_direction": trend_direction,
            "historical_p90": round(historical_p90, 1),
            "shortage_risk": risk,
            "reservoir_pct_full_used": round(reservoir_now, 1),
        }
