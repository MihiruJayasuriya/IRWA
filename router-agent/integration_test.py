"""
Simple integration checks for Person D.

Run after starting Collector, Analysis, IR, and Router:
    python integration_test.py
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8000")
API_KEY = os.getenv("INTERNAL_API_KEY", "water-agent-secret-key")
HEADERS = {"X-API-Key": API_KEY}


def check(name, fn):
    print(f"\n--- {name} ---")
    try:
        result = fn()
        print("PASS")
        print(result)
        return True
    except Exception as exc:
        print("FAIL")
        print(exc)
        return False


def status_check():
    response = httpx.get(f"{ROUTER_URL}/router/status", timeout=10)
    response.raise_for_status()
    data = response.json()
    expected = {"collector", "analysis", "ir", "alert", "summarizer"}
    assert expected.issubset(data.get("services", {}))
    assert all(data["services"][service]["status"] == "ok" for service in expected)
    return {
        service: details.get("status")
        for service, details in data.get("services", {}).items()
    }


def live_analysis_check():
    response = httpx.post(f"{ROUTER_URL}/router/live-analysis", headers=HEADERS, timeout=15)
    response.raise_for_status()
    data = response.json()
    assert data["status"] == "ok"
    assert "reading" in data
    assert "analysis" in data
    assert data["analysis"]["status"] == "ok"
    assert data["summary"]["status"] == "ok"
    assert data["alert_result"]["status"] == "ok"
    return data["route"]


def forecast_check():
    response = httpx.post(
        f"{ROUTER_URL}/router/forecast",
        params={"zone": "West", "days_ahead": 7},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    assert data["status"] == "ok"
    assert data["analysis"]["status"] == "ok"
    assert data["summary"]["status"] == "ok"
    assert data["alert_result"]["status"] == "ok"
    return data["analysis"]["payload"]


def complaint_check():
    body = {
        "agent": "test",
        "type": "citizen_complaint",
        "payload": {"text": "There is a pipe leak in Zone 4 and water is flooding the road."},
    }
    response = httpx.post(f"{ROUTER_URL}/router/process", json=body, headers=HEADERS, timeout=15)
    response.raise_for_status()
    data = response.json()
    assert data["status"] == "ok"
    return data["alert"]


def main():
    checks = [
        check("Router status", status_check),
        check("Live analysis route", live_analysis_check),
        check("Forecast route", forecast_check),
        check("Complaint route", complaint_check),
    ]
    if not all(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
