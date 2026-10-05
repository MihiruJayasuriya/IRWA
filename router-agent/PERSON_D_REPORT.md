# Person D Report - Router Agent and Agent Communication

## Responsibility

Person D is responsible for the Router Agent, the agent communication protocol, and integration between the system's agents.

## What was implemented

- Built a FastAPI Router Agent on port `8000`.
- Added `/router/status` to check whether Collector, Analysis, and IR services are online.
- Added `/router/process` as the main generic routing endpoint.
- Added `/router/live-analysis` for an end-to-end Collector -> Router -> Analysis demo.
- Added `/router/forecast` for demand prediction requests.
- Added `/router/search` for document retrieval requests.
- Added simple complaint routing with urgency classification and zone extraction.
- Added a shared JSON message format for agent communication.
- Added `index.html` as a small Router dashboard for manual testing.
- Added `integration_test.py` for command-line integration checking.

## How the Router Agent works

The Router Agent receives a JSON message from the frontend or another agent. It checks the `type` field, then forwards the request to the correct agent:

- `collector_latest` goes to the Collector Agent.
- `usage_reading` and `forecast_request` go to the Analysis Agent.
- `policy_query` goes to the IR Module.
- `citizen_complaint` is classified by the Router and then sent to the IR Module for supporting documents.

If an analysis result is anomalous, the Router automatically searches the IR Module for leak or abnormal-flow guidance.

## Communication protocol

The system uses REST APIs with JSON messages. Every message follows the same envelope:

```json
{
  "agent": "collector",
  "zone": "West",
  "type": "usage_reading",
  "payload": {},
  "timestamp": "2026-08-14T05:30:00Z"
}
```

Internal agent calls use the `X-API-Key` header. This gives a basic service-to-service security layer for the student demo.

## Why this satisfies the assignment

- Shows agent communication through REST APIs.
- Provides a central orchestration layer.
- Connects AI prediction/anomaly detection with information retrieval.
- Supports NLP-related complaint routing.
- Demonstrates integration readiness for all group members.

## Limitations

- The complaint classifier is rule-based, not a trained NLP model.
- The API key is simple and shared for the demo.
- MCP is represented conceptually by a standard protocol/API layer; the current implementation uses REST because it is easier to integrate for a student system.
