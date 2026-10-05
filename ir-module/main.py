"""
main.py - IR Module API (port 8003).

Endpoints
  GET  /ir/status                open   health check (the Router calls it without a key)
  GET  /ir/search?query=         key    ranked policy passages, with an explanation for each result
  GET  /ir/entities?text=        key    named entities in any text (zones, stages, percentages, durations, ...)
  GET  /ir/documents/{name}      key    full text of one indexed document (for citing / grounding)
  POST /ir/reindex               key    re-read data/documents after files were added or edited

Security
  * The API key comes from the environment (.env), never from source code. Same variable
    as the other agents: INTERNAL_API_KEY (IR_API_KEY overrides it for this service only).
  * Constant-time key comparison. The app refuses to start with a missing, short or
    placeholder key.
  * Queries are cleaned and length-limited; top_k and method are validated.
  * Document names are looked up in the index, never used as file paths, so
    /ir/documents/../main.py cannot read files.

Privacy / Responsible AI
  * Queries can contain personal details (the Router forwards citizen complaints
    here), so query text is NEVER logged or stored: only a short hash, its length,
    and the result count. uvicorn's own access log would print "GET /ir/search?query=..."
    with the full text, so query strings are redacted from it as well.
  * Results say why they matched, and the module abstains instead of returning
    weak matches.
"""

import hashlib
import hmac
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ir_module import IRConfig, WaterPolicyIR
from ner import extract_entities

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ir.api")


class RedactQueryString(logging.Filter):
    """uvicorn's access log records the full request path, query string included.
    Replace everything after '?' so search text never reaches the console or log files."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                a.split("?", 1)[0] + "?[redacted]" if isinstance(a, str) and a.startswith("/") and "?" in a else a
                for a in record.args)
        return True


logging.getLogger("uvicorn.access").addFilter(RedactQueryString())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, "..", "router-agent", ".env"))  # shared key fallback    

# --- Secret management ---------------------------------------------------------------
MIN_KEY_LENGTH = 16
PLACEHOLDER_PREFIX = "change-me"
KNOWN_LEAKED_KEYS = {"water-agent-secret-key"}


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _load_api_key() -> str:
    key = _first_env("IR_API_KEY", "INTERNAL_API_KEY")
    if len(key) < MIN_KEY_LENGTH:
        raise RuntimeError(
            "IR_API_KEY (or INTERNAL_API_KEY) is missing or too short "
            f"(need at least {MIN_KEY_LENGTH} characters). Copy .env.example to .env and set a strong "
            'random key, e.g. python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    if key.lower().startswith(PLACEHOLDER_PREFIX):
        raise RuntimeError(
            "The API key is still the .env.example placeholder. Generate a real one: "
            'python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    return key


INTERNAL_API_KEY = _load_api_key()
if INTERNAL_API_KEY in KNOWN_LEAKED_KEYS:
    log.warning("SECURITY: the API key in use is the old demo key that is in the repository history. "
                "It works, but rotate it in every service's .env before any real deployment.")

ir_engine = WaterPolicyIR(config=IRConfig.from_env())

app = FastAPI(title="IR Module - Water Policy Search", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str = Header(default=None)):
    supplied = (x_api_key or "").encode("utf-8")
    if not hmac.compare_digest(supplied, INTERNAL_API_KEY.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Unauthorized: missing or invalid X-API-Key header")


def clean_query(raw: str) -> str:
    """Remove control characters, collapse whitespace, and enforce non-empty / <= 500 chars
    (the same limit the Router applies)."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", raw or "")).strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="query cannot be empty")
    if len(cleaned) > 500:
        raise HTTPException(status_code=422, detail="query is too long (max 500 characters)")
    return cleaned


@app.get("/ir/status")
def status():
    cfg = ir_engine.cfg
    return {
        "status": "ok",
        "module": "information_retrieval",
        "documents_indexed": len(ir_engine.documents),
        "document_names": ir_engine.doc_names,
        "passages_indexed": len(ir_engine.passages),
        "vocabulary_size": len(ir_engine.vocab),
        "default_method": "bm25",
        "methods": ["bm25", "tfidf"],
        "settings": {"bm25_k1": cfg.k1, "bm25_b": cfg.b, "passage_sentences": cfg.window,
                     "expansion_weight": cfg.expansion_weight, "idf_level": cfg.idf_level},
    }


@app.get("/ir/search")
def search(
    query: str = Query(..., description="Search text, e.g. 'brown water coming from my tap'"),
    top_k: int = Query(3, ge=1, le=10),
    method: str = Query("bm25", pattern="^(bm25|tfidf)$"),
    expand: bool = Query(True, description="Apply the domain thesaurus to the query"),
    _: None = Depends(require_api_key),
):
    q = clean_query(query)
    result = ir_engine.search(q, top_k=top_k, method=method, expand=expand)
    # Privacy: never log the query text (it may contain personal details).
    log.info("search q#%s len=%d method=%s results=%d abstained=%s",
             hashlib.sha256(q.encode("utf-8")).hexdigest()[:8], len(q), method,
             result["num_results"], result["abstained"])
    return {"status": "ok", "query": q, **result}


@app.get("/ir/entities")
def entities(text: str = Query(..., description="Text to scan, e.g. 'reservoir in zone 2 is at 8% for 48 hours'"),
             _: None = Depends(require_api_key)):
    """Named-entity recognition on its own, for other agents (Router, alert agent) that need the
    structured facts in a complaint or an alert message. Same privacy rule as search: never logged."""
    t = clean_query(text)
    found = extract_entities(t)
    log.info("entities q#%s len=%d found=%d", hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], len(t), len(found))
    return {"status": "ok", "text": t, "num_entities": len(found), "entities": found}


@app.get("/ir/documents/{name}")
def get_document(name: str, _: None = Depends(require_api_key)):
    # Looked up in the index, never opened as a path: no directory traversal possible.
    if name not in ir_engine.doc_names:
        raise HTTPException(status_code=404, detail="Document not found")
    i = ir_engine.doc_names.index(name)
    return {"status": "ok", "document": name, "title": ir_engine.titles[i], "text": ir_engine.documents[i]}


@app.post("/ir/reindex")
def reindex(_: None = Depends(require_api_key)):
    count = ir_engine.reload()
    log.info("Reindexed: %d documents, %d passages", count, len(ir_engine.passages))
    return {"status": "ok", "documents_indexed": count, "passages_indexed": len(ir_engine.passages),
            "document_names": ir_engine.doc_names}
