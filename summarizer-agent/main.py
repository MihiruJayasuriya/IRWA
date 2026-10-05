import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

load_dotenv(Path(__file__).resolve().parents[1] / "router-agent" / ".env")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "water-agent-secret-key")

from logic import create_summary


app = FastAPI(
    title="Water Management - Summarizer Agent",
    description="Summarizes analysis and forecast results from the Analysis Agent.",
    version="1.0.0"
)


# ============================================================
# Request Validation Model
# ============================================================

class SummarizerRequest(BaseModel):
    status: str = "ok"

    agent: str = Field(
        min_length=1,
        max_length=30
    )

    type: str = Field(
        min_length=1,
        max_length=50
    )

    zone: str = Field(
        min_length=1,
        max_length=50
    )

    payload: Dict[str, Any]

    timestamp: Optional[str] = None


# ============================================================
# Health / Status Endpoint
# ============================================================

@app.get("/summarizer/status")
def status():
    return {
        "status": "ok",
        "agent": "summarizer"
    }


# ============================================================
# Main Summarizer Endpoint
# ============================================================

def require_api_key(x_api_key: str = Header(default=None)):
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/summarizer/process")
def process(data: SummarizerRequest, _: None = Depends(require_api_key)):

    try:

        # ----------------------------------------------------
        # 1. Validate agent
        # ----------------------------------------------------

        if data.agent != "analysis":
            raise HTTPException(
                status_code=400,
                detail="Invalid source agent. Expected 'analysis'."
            )

        # ----------------------------------------------------
        # 2. Validate message type
        # ----------------------------------------------------

        allowed_types = {
            "analysis_result",
            "forecast_result"
        }

        if data.type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Unsupported analysis result type."
            )

        # ----------------------------------------------------
        # 3. Validate zone
        # ----------------------------------------------------

        zone = data.zone.strip()

        if not zone:
            raise HTTPException(
                status_code=400,
                detail="Zone cannot be empty."
            )

        # ----------------------------------------------------
        # 4. Validate payload
        # ----------------------------------------------------

        if not data.payload:
            raise HTTPException(
                status_code=400,
                detail="Payload cannot be empty."
            )

        # ----------------------------------------------------
        # 5. Validate water usage
        # ----------------------------------------------------

        value = data.payload.get("value")

        if value is not None:

            if not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail="Water usage value must be numeric."
                )

            if value < 0:
                raise HTTPException(
                    status_code=400,
                    detail="Water usage value cannot be negative."
                )

        # ----------------------------------------------------
        # 6. Validate reservoir percentage
        # ----------------------------------------------------

        reservoir = data.payload.get(
            "reservoir_pct_full_used"
        )

        if reservoir is not None:

            if not isinstance(reservoir, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail="Reservoir percentage must be numeric."
                )

            if not 0 <= reservoir <= 100:
                raise HTTPException(
                    status_code=400,
                    detail="Reservoir percentage must be between 0 and 100."
                )

        # ----------------------------------------------------
        # 7. Prepare validated data
        # ----------------------------------------------------

        validated_data = data.model_dump()

        validated_data["zone"] = zone

        # ----------------------------------------------------
        # 8. Generate summary
        # ----------------------------------------------------

        summary = create_summary(validated_data)

        # ----------------------------------------------------
        # 9. Return standard response
        # ----------------------------------------------------

        return {
            "status": "ok",
            "agent": "summarizer",
            "type": "summary_result",
            "zone": zone,
            "payload": {
                "summary": summary
            },
            "timestamp": data.timestamp
        }

    except HTTPException:
        raise

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:

        # Do not expose internal Python errors to users
        raise HTTPException(
            status_code=500,
            detail="Unable to process the summarization request."
        )
