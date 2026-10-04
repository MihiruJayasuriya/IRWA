"""
Analysis Agent - FastAPI entrypoint (v3, uses rainfall + reservoir features)
Port: 8002
Endpoints: POST /analysis/process, GET /analysis/status
Follows the group's shared message envelope:
{ "agent": ..., "zone": ..., "type": ..., "payload": {...}, "timestamp": ... }
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from logic import AnalysisEngine

load_dotenv()

PORT = int(os.getenv("PORT", 8002))

app = FastAPI(title="Analysis Agent")
engine: Optional[AnalysisEngine] = None


class AgentMessage(BaseModel):
    agent: str
    zone: Optional[str] = None
    type: str
    payload: Dict[str, Any] = {}
    timestamp: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.on_event("startup")
def startup():
    global engine
    engine = AnalysisEngine()
    print(f"[Analysis Agent] loaded trained model — zones: {engine.zones} "
          f"— environmental features: {engine.uses_environmental}")


@app.get("/analysis/status")
def status():
    return {
        "status": "ok",
        "agent": "analysis",
        "zones_loaded": engine.zones if engine else [],
        "model_loaded": engine is not None,
        "uses_environmental_features": engine.uses_environmental if engine else False,
        "timestamp": now_iso(),
    }


@app.post("/analysis/process")
def process(message: AgentMessage):
    if engine is None:
        return {"status": "error", "agent": "analysis", "error": "engine not initialised", "timestamp": now_iso()}

    try:
        if message.type == "usage_reading":
            zone = message.zone or message.payload.get("zone")
            value = message.payload.get("consumption_liters")

            if zone is None or value is None:
                return {
                    "status": "error",
                    "agent": "analysis",
                    "error": "usage_reading requires zone and payload.consumption_liters",
                    "timestamp": now_iso(),
                }

            result = engine.score_reading(zone, float(value))
            return {
                "status": "ok",
                "agent": "analysis",
                "type": "analysis_result",
                "zone": zone,
                "payload": result,
                "timestamp": now_iso(),
            }

        elif message.type == "forecast_request":
            zone = message.zone or message.payload.get("zone")
            days_ahead = int(message.payload.get("days_ahead", 7))
            rainfall_mm = message.payload.get("rainfall_mm")
            reservoir_pct_full = message.payload.get("reservoir_pct_full")

            if zone is None:
                return {
                    "status": "error",
                    "agent": "analysis",
                    "error": "forecast_request requires zone",
                    "timestamp": now_iso(),
                }

            result = engine.forecast_zone(
                zone, days_ahead,
                rainfall_mm=rainfall_mm,
                reservoir_pct_full=reservoir_pct_full,
            )
            return {
                "status": "ok",
                "agent": "analysis",
                "type": "forecast_result",
                "zone": zone,
                "payload": result,
                "timestamp": now_iso(),
            }

        else:
            return {
                "status": "error",
                "agent": "analysis",
                "error": f"unknown message type '{message.type}'. Expected 'usage_reading' or 'forecast_request'.",
                "timestamp": now_iso(),
            }

    except Exception as e:
        return {"status": "error", "agent": "analysis", "error": str(e), "timestamp": now_iso()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
