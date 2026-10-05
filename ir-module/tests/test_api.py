"""Tests for the HTTP service (main.py): authentication, validation, privacy."""
import logging
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

import main

KEY = os.environ["INTERNAL_API_KEY"]
H = {"X-API-Key": KEY}
client = TestClient(main.app)


def test_status_is_open_and_keeps_the_fields_the_router_reads():
    d = client.get("/ir/status").json()
    assert d["status"] == "ok" and d["documents_indexed"] == 11 and len(d["document_names"]) == 11
    assert d["passages_indexed"] > 11 and "bm25" in d["methods"]


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}, {"X-API-Key": "kéy".encode("utf-8")}])
def test_search_requires_the_key(headers):
    assert client.get("/ir/search", params={"query": "leak"}, headers=headers).status_code == 401


def test_the_old_hardcoded_key_no_longer_works():
    assert client.get("/ir/search", params={"query": "leak"}, headers={"X-API-Key": "water-agent-secret-key"}).status_code == 401


def test_search_returns_explained_results():
    r = client.get("/ir/search", params={"query": "brown water coming from my tap"}, headers=H).json()
    assert r["status"] == "ok" and r["num_results"] == len(r["results"]) >= 1
    assert {"document", "score", "snippet", "matched_terms", "passage_id"} <= set(r["results"][0])
    assert r["abstained"] is False and r["expanded_terms"]


def test_out_of_scope_query_abstains_over_http():
    r = client.get("/ir/search", params={"query": "what is the capital of france"}, headers=H).json()
    assert r["status"] == "ok" and r["results"] == [] and r["abstained"] is True


@pytest.mark.parametrize("params", [
    {"query": ""}, {"query": "   "}, {"query": "x" * 501},
    {"query": "leak", "top_k": 0}, {"query": "leak", "top_k": 11}, {"query": "leak", "method": "sql"},
])
def test_bad_input_is_rejected(params):
    assert client.get("/ir/search", params=params, headers=H).status_code == 422


def test_missing_query_is_rejected():
    assert client.get("/ir/search", headers=H).status_code == 422


def test_control_characters_are_stripped():
    r = client.get("/ir/search", params={"query": "drought\x00 stage\x07 restrictions"}, headers=H)
    assert r.status_code == 200 and "\x00" not in r.json()["query"]


def test_query_text_is_never_logged(caplog):
    """Citizen complaints (possibly with names / addresses) reach this service."""
    secret = "zzprivatemarker4711 lives at 12 Example Road"
    with caplog.at_level(logging.DEBUG, logger="ir"):
        client.get("/ir/search", params={"query": secret + " leak"}, headers=H)
    assert "zzprivatemarker4711" not in caplog.text and "Example Road" not in caplog.text
    assert "search q#" in caplog.text                          # ...but the request itself is audited


def test_uvicorn_access_log_query_strings_are_redacted():
    """Regression: uvicorn prints 'GET /ir/search?query=<full complaint text>' by default."""
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                               ("127.0.0.1:5000", "GET", "/ir/search?query=my+name+is+Jane+Doe", "1.1", 200), None)
    flt = logging.getLogger("uvicorn.access").filters
    assert any(isinstance(f, main.RedactQueryString) for f in flt)
    main.RedactQueryString().filter(record)
    line = record.getMessage()
    assert "Jane" not in line and "/ir/search?[redacted]" in line
    plain = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s "%s"', ("a", "/ir/status"), None)
    main.RedactQueryString().filter(plain)
    assert plain.getMessage() == 'a "/ir/status"'                # untouched when there is no query string


def test_document_endpoint_serves_only_indexed_documents():
    ok = client.get("/ir/documents/drought_response_plan.txt", headers=H)
    assert ok.status_code == 200 and "Stage 1" in ok.json()["text"]
    assert client.get("/ir/documents/nope.txt", headers=H).status_code == 404
    for evil in ["..%2Fmain.py", "..%5Cmain.py", "%2e%2e%2fmain.py", "main.py"]:
        assert client.get(f"/ir/documents/{evil}", headers=H).status_code == 404
    assert client.get("/ir/documents/drought_response_plan.txt").status_code == 401


def test_reindex_requires_the_key_and_works():
    assert client.post("/ir/reindex").status_code == 401
    r = client.post("/ir/reindex", headers=H).json()
    assert r["status"] == "ok" and r["documents_indexed"] == 11


# ------------------------------------------------------------ entities endpoint
def test_entities_endpoint_requires_the_key_and_returns_normalised_facts():
    assert client.get("/ir/entities", params={"text": "stage 3 in zone 2"}).status_code == 401
    r = client.get("/ir/entities", params={"text": "Stage 3 restrictions in zone 2 within 48 hours"}, headers=H).json()
    assert r["status"] == "ok" and r["num_entities"] == len(r["entities"]) == 3
    assert {(e["type"], e["normalized"]) for e in r["entities"]} == {("DROUGHT_STAGE", "3"), ("ZONE", "2"), ("DURATION", "48 hour")}


@pytest.mark.parametrize("text", ["", "   ", "x" * 501])
def test_entities_endpoint_validates_input(text):
    assert client.get("/ir/entities", params={"text": text}, headers=H).status_code == 422
    assert client.get("/ir/entities", headers=H).status_code == 422


def test_entities_endpoint_never_logs_the_text(caplog):
    with caplog.at_level(logging.DEBUG, logger="ir"):
        client.get("/ir/entities", params={"text": "zzprivatemarker9931 lives in zone 4"}, headers=H)
    assert "zzprivatemarker9931" not in caplog.text and "entities q#" in caplog.text


def test_search_results_carry_entities_over_http():
    r = client.get("/ir/search", params={"query": "Stage 3 emergency restrictions"}, headers=H).json()
    assert ("DROUGHT_STAGE", "3") in {(e["type"], e["normalized"]) for e in r["entities"]}
    assert "entities" in r["results"][0] and "entity_matches" in r["results"][0]


# ------------------------------------------------------------ startup guards
def _import_main(env):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = {k: v for k, v in os.environ.items() if k not in ("INTERNAL_API_KEY", "IR_API_KEY")}
    full.update(env)
    return subprocess.run([sys.executable, "-W", "ignore", "-c", "import main"], cwd=root, env=full,
                          capture_output=True, text=True)


def test_refuses_to_start_with_a_short_key():
    """(An explicit env var wins over any .env file, so this holds on every machine.)"""
    out = _import_main({"INTERNAL_API_KEY": "short"})
    assert out.returncode != 0 and "too short" in out.stderr


def test_refuses_the_env_example_placeholder():
    out = _import_main({"INTERNAL_API_KEY": "change-me-to-a-long-random-string"})
    assert out.returncode != 0 and "placeholder" in out.stderr


def test_warns_when_the_old_leaked_key_is_used():
    out = _import_main({"INTERNAL_API_KEY": "water-agent-secret-key"})
    assert out.returncode == 0 and "SECURITY" in out.stderr
