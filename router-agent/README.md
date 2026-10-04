# Router Agent

Central coordinator for the Smart Water Management multi-agent system.

## Purpose

The Router Agent receives a standard agent message, decides which service should handle it, calls the correct downstream agent, and returns one combined JSON response.

It connects:

- Collector Agent: `http://localhost:8001`
- Analysis Agent: `http://localhost:8002`
- IR Module: `http://localhost:8003`
- Alert Agent: `http://localhost:8004`
- Summarizer Agent: `http://localhost:8005`
- Router Agent: `http://localhost:8000`

Open `http://localhost:8000/` in a browser for the integrated Waterwise dashboard. It shows the status of all six agents and provides live analysis, forecasts, summaries, alerts, complaint routing, and guidance search.

## Message format

```json
{
  "agent": "router",
  "zone": "West",
  "type": "usage_reading",
  "payload": {
    "consumption_liters": 18000
  },
  "timestamp": "2026-08-14T05:30:00Z"
}
```

## Supported routes

- `collector_latest`: gets the next live reading from Collector.
- `usage_reading`: sends a reading to Analysis, then its result to Summarizer and Alert. If anomalous, also searches IR documents.
- `forecast_request`: sends a forecast request to Analysis, then its result to Summarizer and Alert.
- `policy_query`: searches IR documents.
- `citizen_complaint`: classifies complaint urgency, extracts a simple zone mention, and searches IR documents.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Start the other services first:

```bash
# collector-agent
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# iwra/analysis-agent
uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# ir-module
uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

## Example curl

```bash
curl -X POST "http://localhost:8000/router/process" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: water-agent-secret-key" \
  -d "{\"agent\":\"frontend\",\"zone\":\"West\",\"type\":\"forecast_request\",\"payload\":{\"days_ahead\":7},\"timestamp\":\"2026-08-14T05:30:00Z\"}"
```
