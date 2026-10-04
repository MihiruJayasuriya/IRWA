# Agent Communication Protocol

Person D owns this standard so all agents can connect without changing each other's code.

## Transport

- Protocol: REST over HTTP
- Data format: JSON
- Auth header for internal services: `X-API-Key`
- Timestamp standard: ISO 8601 UTC, for example `2026-08-14T05:30:00Z`

## Ports

| Service | Port | Base URL |
| --- | ---: | --- |
| Router Agent | 8000 | `http://localhost:8000` |
| Collector Agent | 8001 | `http://localhost:8001` |
| Analysis Agent | 8002 | `http://localhost:8002` |
| IR Module | 8003 | `http://localhost:8003` |
| Alert Agent | 8004 | `http://localhost:8004` |
| Summarizer Agent | 8005 | `http://localhost:8005` |

## Standard message envelope

All agent-to-agent requests should use this shape when possible:

```json
{
  "agent": "collector",
  "zone": "West",
  "type": "usage_reading",
  "payload": {
    "consumption_liters": 18000,
    "rainfall_mm": 2.5,
    "reservoir_pct_full": 42.0
  },
  "timestamp": "2026-08-14T05:30:00Z"
}
```

## Standard response shape

All services should return JSON with at least:

```json
{
  "status": "ok",
  "agent": "router",
  "type": "analysis_result",
  "payload": {},
  "timestamp": "2026-08-14T05:30:00Z"
}
```

For failures:

```json
{
  "status": "error",
  "agent": "router",
  "error": "short explanation",
  "timestamp": "2026-08-14T05:30:00Z"
}
```

## Router routes

| Message type | Routed to | Purpose |
| --- | --- | --- |
| `collector_latest` | Collector | Get the next simulated live sensor reading |
| `usage_reading` | Analysis, Summarizer, Alert, optionally IR | Detect anomalies, summarize the result, generate an alert, and retrieve supporting documents when needed |
| `forecast_request` | Analysis, Summarizer, Alert | Predict demand, summarize the forecast, and check shortage risk |
| `policy_query` | IR | Retrieve relevant policy/procedure documents |
| `citizen_complaint` | Router NLP, IR | Classify complaint urgency, extract zone, retrieve guidance |

## Integration rule

Other agents should not call each other directly from the frontend. The frontend should call the Router Agent, then the Router Agent should call the correct service.
