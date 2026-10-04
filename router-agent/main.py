"""
Router Agent - central coordinator for the water management system.

Port: 8000
Endpoints:
  GET  /router/status
  POST /router/process
  POST /router/live-analysis
  POST /router/forecast
  POST /router/search

The router accepts the group's shared message envelope and forwards work
to Collector, Analysis, and IR services.
"""

import os
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from logic import (
    build_anomaly_ir_query,
    build_service_headers,
    classify_complaint,
    clean_text,
    extract_zone,
    now_iso,
    should_attach_ir_evidence,
)

load_dotenv(Path(__file__).with_name(".env"))

PORT = int(os.getenv("PORT", 8000))
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "water-agent-secret-key")
COLLECTOR_URL = os.getenv("COLLECTOR_URL", "http://localhost:8001")
ANALYSIS_URL = os.getenv("ANALYSIS_URL", "http://localhost:8002")
IR_URL = os.getenv("IR_URL", "http://localhost:8003")
ALERT_URL = os.getenv("ALERT_URL", "http://localhost:8004")
SUMMARIZER_URL = os.getenv("SUMMARIZER_URL", "http://localhost:8005")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 8))

app = FastAPI(title="Router Agent", version="1.0.0")


@app.get("/", include_in_schema=False)
def dashboard():
    page = FileResponse(Path(__file__).with_name("index.html"))
    page.set_cookie("waterwise_session", INTERNAL_API_KEY, httponly=True, samesite="strict")
    return page

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentMessage(BaseModel):
    agent: str = Field(default="router")
    zone: Optional[str] = None
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None


def headers() -> Dict[str, str]:
    return build_service_headers(INTERNAL_API_KEY)


def require_api_key(x_api_key: str = Header(default=None), waterwise_session: str = Cookie(default=None)):
    if x_api_key != INTERNAL_API_KEY and waterwise_session != INTERNAL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: missing or invalid X-API-Key header",
        )


async def get_json(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    response = await client.get(url, headers=headers())
    response.raise_for_status()
    return response.json()


async def post_json(client: httpx.AsyncClient, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    response = await client.post(url, json=body, headers=headers())
    response.raise_for_status()
    return response.json()


async def search_ir(client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    response = await client.get(f"{IR_URL}/ir/search", params={"query": query}, headers=headers())
    response.raise_for_status()
    return response.json()


async def enrich_analysis(client: httpx.AsyncClient, analysis: Dict[str, Any]) -> Dict[str, Any]:
    if analysis.get("status") != "ok":
        return {"summary": None, "alert_result": None}

    async def safely_post(url: str) -> Dict[str, Any]:
        try:
            return await post_json(client, url, analysis)
        except (httpx.HTTPError, ValueError) as exc:
            return {"status": "error", "error": str(exc)}

    summary, alert_result = await asyncio.gather(
        safely_post(f"{SUMMARIZER_URL}/summarizer/process"),
        safely_post(f"{ALERT_URL}/alert/process"),
    )
    return {"summary": summary, "alert_result": alert_result}


@app.get("/router/status")
async def status():
    service_urls = {
        "collector": f"{COLLECTOR_URL}/collector/status",
        "analysis": f"{ANALYSIS_URL}/analysis/status",
        "ir": f"{IR_URL}/ir/status",
        "alert": f"{ALERT_URL}/alert/status",
        "summarizer": f"{SUMMARIZER_URL}/summarizer/status",
    }
    services: Dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for name, url in service_urls.items():
            try:
                response = await client.get(url)
                services[name] = {
                    "status": "ok" if response.is_success else "error",
                    "http_status": response.status_code,
                    "details": response.json() if response.headers.get("content-type", "").startswith("application/json") else None,
                }
            except Exception as exc:
                services[name] = {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "agent": "router",
        "services": services,
        "timestamp": now_iso(),
    }


@app.post("/router/process")
async def process(message: AgentMessage, _: None = Depends(require_api_key)):
    """
    Generic routing endpoint.

    Supported message.type values:
      - usage_reading       -> Analysis Agent + optional IR recommendation
      - forecast_request    -> Analysis Agent
      - policy_query        -> IR Module
      - citizen_complaint   -> simple NLP classification + IR Module
      - collector_latest    -> Collector Agent
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            if message.type == "collector_latest":
                collector = await get_json(client, f"{COLLECTOR_URL}/collector/latest")
                return {"status": "ok", "agent": "router", "route": ["collector"], "result": collector, "timestamp": now_iso()}

            if message.type in {"usage_reading", "forecast_request"}:
                analysis = await post_json(client, f"{ANALYSIS_URL}/analysis/process", message.model_dump())
                route = ["analysis"]
                enrichment = await enrich_analysis(client, analysis)
                route.extend(["summarizer", "alert"])
                evidence = None

                payload = analysis.get("payload", {})
                if should_attach_ir_evidence(payload):
                    query = build_anomaly_ir_query(message.zone, payload)
                    evidence = await search_ir(client, query)
                    route.append("ir")

                return {
                    "status": "ok",
                    "agent": "router",
                    "route": route,
                    "analysis": analysis,
                    **enrichment,
                    "supporting_documents": evidence,
                    "timestamp": now_iso(),
                }

            if message.type == "policy_query":
                query = clean_text(message.payload.get("query"), "payload.query")
                evidence = await search_ir(client, query)
                return {"status": "ok", "agent": "router", "route": ["ir"], "result": evidence, "timestamp": now_iso()}

            if message.type == "citizen_complaint":
                text = clean_text(message.payload.get("text"), "payload.text")
                classification = classify_complaint(text)
                zone = message.zone or extract_zone(text)
                evidence = await search_ir(client, text)
                alert = {
                    "zone": zone,
                    "category": classification["category"],
                    "urgency": classification["urgency"],
                    "message": text,
                    "recommended_action": "Review immediately" if classification["urgency"] == "high" else "Add to staff queue",
                }
                return {
                    "status": "ok",
                    "agent": "router",
                    "route": ["nlp_classifier", "ir"],
                    "type": "complaint_routing_result",
                    "alert": alert,
                    "supporting_documents": evidence,
                    "timestamp": now_iso(),
                }

            return {
                "status": "error",
                "agent": "router",
                "error": f"unknown message type '{message.type}'",
                "timestamp": now_iso(),
            }
        except httpx.HTTPStatusError as exc:
            return {
                "status": "error",
                "agent": "router",
                "error": f"downstream service returned HTTP {exc.response.status_code}",
                "details": exc.response.text,
                "timestamp": now_iso(),
            }
        except httpx.RequestError as exc:
            return {
                "status": "error",
                "agent": "router",
                "error": f"could not reach downstream service: {exc}",
                "timestamp": now_iso(),
            }


@app.post("/router/live-analysis")
async def live_analysis(_: None = Depends(require_api_key)):
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        reading = await get_json(client, f"{COLLECTOR_URL}/collector/latest")
        analysis = await post_json(client, f"{ANALYSIS_URL}/analysis/process", reading)
        enrichment = await enrich_analysis(client, analysis)
        return {
            "status": "ok",
            "agent": "router",
            "route": ["collector", "analysis", "summarizer", "alert"],
            "reading": reading,
            "analysis": analysis,
            **enrichment,
            "timestamp": now_iso(),
        }


@app.post("/router/forecast")
async def forecast(
    zone: str = Query(...),
    days_ahead: int = Query(7, ge=1, le=30),
    _: None = Depends(require_api_key),
):
    message = {
        "agent": "router",
        "zone": zone,
        "type": "forecast_request",
        "payload": {"days_ahead": days_ahead},
        "timestamp": now_iso(),
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        analysis = await post_json(client, f"{ANALYSIS_URL}/analysis/process", message)
        enrichment = await enrich_analysis(client, analysis)
    return {"status": "ok", "agent": "router", "route": ["analysis", "summarizer", "alert"], "analysis": analysis, **enrichment, "timestamp": now_iso()}


@app.post("/router/search")
async def search(
    query: str = Query(...),
    _: None = Depends(require_api_key),
):
    query = clean_text(query, "query")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        evidence = await search_ir(client, query)
    return {"status": "ok", "agent": "router", "route": ["ir"], "result": evidence, "timestamp": now_iso()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
