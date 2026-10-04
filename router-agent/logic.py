"""
Router Agent shared logic.

These helpers are kept separate from main.py so Person D's folder follows
the same main.py / logic.py pattern as the other agents.
"""

import re
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import HTTPException


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(value: str, field_name: str = "text") -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail=f"{field_name} cannot be empty")
    if len(cleaned) > 500:
        raise HTTPException(status_code=422, detail=f"{field_name} is too long")
    return cleaned


def extract_zone(text: str) -> Optional[str]:
    match = re.search(r"\b(?:zone|area|region)\s*([A-Za-z]+|\d+)\b", text, re.IGNORECASE)
    if not match:
        return None

    zone = match.group(1)
    zone_map = {"1": "North", "2": "South", "3": "East", "4": "West", "5": "Central"}
    return zone_map.get(zone, zone.title())


def classify_complaint(text: str) -> Dict[str, str]:
    lower = text.lower()
    urgent_terms = ["leak", "burst", "contamination", "dirty", "smell", "no water", "flood"]
    drought_terms = ["shortage", "low pressure", "dry", "reservoir", "ration"]
    billing_terms = ["bill", "charge", "meter", "usage"]

    if any(term in lower for term in urgent_terms):
        return {"category": "incident", "urgency": "high"}
    if any(term in lower for term in drought_terms):
        return {"category": "shortage_risk", "urgency": "medium"}
    if any(term in lower for term in billing_terms):
        return {"category": "billing_or_usage", "urgency": "normal"}
    return {"category": "general", "urgency": "normal"}


def build_service_headers(api_key: str) -> Dict[str, str]:
    return {"X-API-Key": api_key}


def should_attach_ir_evidence(analysis_payload: Dict) -> bool:
    return bool(analysis_payload.get("is_anomaly"))


def build_anomaly_ir_query(zone: Optional[str], analysis_payload: Dict) -> str:
    result_zone = zone or analysis_payload.get("zone") or "unknown zone"
    severity = analysis_payload.get("severity", "unknown")
    return f"{result_zone} leak abnormal flow {severity}"
