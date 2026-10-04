
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from logic import get_historical_baseline, get_next_reading, get_summary

app = FastAPI(title="Collector Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In a real deployment this would come from an environment variable,
# not be hardcoded. Hardcoded here for a student project demo -
# worth noting explicitly in the report/viva rather than presenting
# it as production-grade secret management.
load_dotenv(Path(__file__).resolve().parents[1] / "router-agent" / ".env")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "water-agent-secret-key")


def require_api_key(x_api_key: str = Header(default=None)):
    """Dependency that actually enforces the API key - unlike a bare
    function definition, wiring this via Depends() on a route means
    FastAPI runs it before the route body executes."""
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: missing or invalid X-API-Key header",
        )


def validate_region(region: str) -> str:
    """Basic input sanitization: reject empty, overly long, or
    non-alphabetic region names before they reach the database query."""
    cleaned = region.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Region cannot be empty")
    if len(cleaned) > 50:
        raise HTTPException(status_code=422, detail="Region name too long")
    if not re.match(r"^[A-Za-z ]+$", cleaned):
        raise HTTPException(status_code=422, detail="Region must contain only letters and spaces")
    return cleaned


@app.get("/collector/status")
def status():
    # Left open intentionally - a health check endpoint is generally
    # safe to expose for monitoring/uptime checks.
    return {
        "status": "ok",
        "agent": "collector",
        "dataset_summary": get_summary(),
    }


@app.get("/collector/latest")
def latest(_: None = Depends(require_api_key)):
    return get_next_reading()


@app.get("/collector/history/{region}")
def history(region: str, _: None = Depends(require_api_key)):
    clean_region = validate_region(region)
    baseline = get_historical_baseline(clean_region)
    if baseline is None:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for region '{clean_region}'",
        )
    return {"status": "ok", "agent": "collector", "baseline": baseline}
