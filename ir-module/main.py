"""
main.py - IR Module standalone FastAPI app

Exposes the WaterPolicyIR search engine as a small API, separate from
Collector Agent, so it can be demoed on its own and handed off to
whichever agent (Analysis/Router) ends up owning it.

Security model (same as Collector Agent, for consistency across the
system): Router Agent is expected to handle perimeter/user-facing
security. This agent additionally protects itself at the service
level via an X-API-Key header check, so it isn't wide open if
something bypasses Router and calls this port directly.

Endpoints:
  GET /ir/status        -> health check (left open - safe for monitoring)
  GET /ir/search?query=  -> TF-IDF search over the policy documents (requires API key)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from ir_module import WaterPolicyIR

app = FastAPI(title="IR Module - Water Policy Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ir_engine = WaterPolicyIR()

load_dotenv(Path(__file__).resolve().parents[1] / "router-agent" / ".env")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "water-agent-secret-key")


def require_api_key(x_api_key: str = Header(default=None)):
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: missing or invalid X-API-Key header",
        )


@app.get("/ir/status")
def status():
    return {
        "status": "ok",
        "module": "information_retrieval",
        "documents_indexed": len(ir_engine.documents),
        "document_names": ir_engine.doc_names,
    }


@app.get("/ir/search")
def search(
    query: str = Query(..., description="Search text, e.g. 'reservoir low drought'"),
    _: None = Depends(require_api_key),
):
    results = ir_engine.search(query, top_k=3)
    return {
        "status": "ok",
        "query": query,
        "num_results": len(results),
        "results": results,
    }
