import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from logic import create_alert

load_dotenv(Path(__file__).resolve().parents[1] / "router-agent" / ".env")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "water-agent-secret-key")


app = FastAPI(
    title="Water Management Alert Agent",
    description="Alert Agent for the Water Management System",
    version="1.0.0"
)


@app.get("/alert/status")
def status():
    return {
        "status": "ok",
        "agent": "alert"
    }


def require_api_key(x_api_key: str = Header(default=None)):
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/alert/process")
def process_alert(data: dict, _: None = Depends(require_api_key)):
    return create_alert(data)
