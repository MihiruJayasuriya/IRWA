"""Tests for the autonomous loop (agent.py) against a fake Router - no real network."""
import asyncio
import json

import httpx

from agent import AgentConfig, CollectorAgent
from rules import RuleConfig

KEY = "test-key-for-unit-tests-0123456789"


# ----------------------------------------------------------------- fixtures
class FakeStream:
    def __init__(self, readings):
        self.readings, self.i = readings, 0

    def next(self, consumer="x"):
        r = dict(self.readings[self.i % len(self.readings)])
        r.setdefault("sequence_id", self.i + 1)
        r.setdefault("cycle", 0)
        self.i += 1
        return r


def mk(day, zone="South", c=10000.0, rain=0.0, res=60.0, **extra):
    return {"zone": zone, "date": f"2023-06-{day:02d}", "consumption_liters": c,
            "rainfall_mm": rain, "reservoir_pct_full": res, **extra}


def seed(zone="South", reservoir=60.0):
    return {zone: [{"date": f"2023-05-{15 + i:02d}", "consumption_liters": 10000.0,
                    "rainfall_mm": 0.0, "reservoir_pct_full": reservoir + i * 0.7} for i in range(10)]}


class FakeRouter:
    """Mimics the Router's /router/process behaviour for the message types the collector uses."""

    def __init__(self, anomaly=lambda msg: False, down=False, error_status=False):
        self.calls, self.anomaly, self.down, self.error_status = [], anomaly, down, error_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("router is down", request=request)
        msg = json.loads(request.content)
        self.calls.append({"msg": msg, "key": request.headers.get("x-api-key")})
        if self.error_status:
            return httpx.Response(200, json={"status": "error", "agent": "router", "error": "analysis offline"})
        t = msg["type"]
        if t == "usage_reading":
            anomalous = self.anomaly(msg)
            return httpx.Response(200, json={"status": "ok", "agent": "router", "route": ["analysis"],
                "analysis": {"status": "ok", "payload": {
                    "is_anomaly": anomalous, "z_score": 3.4 if anomalous else 0.2,
                    "severity": "high" if anomalous else "normal",
                    "baseline_mean": 10000.0, "baseline_std": 1500.0}},
                "supporting_documents": None})
        if t == "policy_query":
            return httpx.Response(200, json={"status": "ok", "agent": "router", "route": ["ir"],
                "result": {"results": [{"document": "leak_detection_procedure.txt", "score": 0.41,
                                        "snippet": "A suspected leak is flagged when..."}]}})
        if t == "forecast_request":
            return httpx.Response(200, json={"status": "ok", "agent": "router", "route": ["analysis"],
                "analysis": {"status": "ok", "payload": {"days_ahead": 7, "shortage_risk": "high",
                    "trend_direction": "rising", "forecast": [1, 2, 3], "historical_p90": 13000.0}}})
        return httpx.Response(200, json={"status": "error", "error": "unknown type"})

    def types(self):
        return [c["msg"]["type"] for c in self.calls]


def build(readings, router, zone_seed=None, **cfg):
    config = AgentConfig(api_key=KEY, poll_seconds=0.01, min_poll_seconds=0.001, **cfg)
    return CollectorAgent(FakeStream(readings), lambda: zone_seed or seed(), RuleConfig(), config,
                          transport=httpx.MockTransport(router.handler))


def run_ticks(agent, n):
    async def go():
        agent._client = httpx.AsyncClient(transport=agent._transport)
        try:
            return [await agent.tick() for _ in range(n)]
        finally:
            await agent._client.aclose()
    return asyncio.run(go())


# -------------------------------------------------------------------- tests
def test_normal_readings_are_forwarded_with_assessment_and_provenance():
    router = FakeRouter()
    agent = build([mk(1), mk(2)], router)
    run_ticks(agent, 2)
    assert router.types() == ["usage_reading", "usage_reading"]
    call = router.calls[0]
    assert call["key"] == KEY                                   # authenticates to the router
    msg = call["msg"]
    assert msg["agent"] == "collector" and msg["zone"] == "South"
    assert msg["payload"]["collector_assessment"]["quality"] == "ok"
    assert msg["payload"]["provenance"]["source"] == "replayed_historical_dataset"
    assert msg["timestamp"].endswith("Z")                       # group protocol timestamp
    assert agent.forwarded == 2 and agent.get_alerts() == []


def test_leak_is_detected_confirmed_and_enriched_with_sop_evidence():
    router = FakeRouter()
    agent = build([mk(1, c=15000), mk(2, c=15000), mk(3, c=15000)], router)
    run_ticks(agent, 3)
    alerts = agent.get_alerts()
    assert [(a["category"], a["event"]) for a in alerts] == [("leak", "suspected"), ("leak", "confirmed")]
    a = alerts[1]
    assert a["rule_id"] == "LEAK-01" and a["source_document"] == "leak_detection_procedure.txt"
    assert a["requires_human_review"] and a["evidence"][0]["document"] == "leak_detection_procedure.txt"
    assert a["enrichment"]["status"] == "complete"
    assert "policy_query" in router.types()
    assert agent.mode == "heightened"                            # adaptive sampling kicked in


def test_drought_escalation_requests_forecast_and_sop():
    router = FakeRouter()
    z = seed(reservoir=32)
    agent = build([mk(1, res=35), mk(2, res=28)], router, zone_seed=z)
    run_ticks(agent, 2)
    (alert,) = agent.get_alerts()
    assert (alert["category"], alert["event"], alert["severity"]) == ("drought", "escalated", "warning")
    assert alert["forecast"]["shortage_risk"] == "high"
    assert alert["evidence"] and alert["enrichment"]["status"] == "complete"
    assert "forecast_request" in router.types() and "policy_query" in router.types()
    # forecast is seeded with the CURRENT readings so Analysis has context
    fc = [c["msg"] for c in router.calls if c["msg"]["type"] == "forecast_request"][0]
    assert fc["payload"]["reservoir_pct_full"] == 28.0


def test_invalid_reading_is_quarantined_not_forwarded():
    router = FakeRouter()
    agent = build([mk(1, c=-50), mk(2, c=-50), mk(3)], router)
    run_ticks(agent, 3)
    assert agent.quarantined == 2 and agent.forwarded == 1
    assert router.types() == ["usage_reading"]                  # the bad rows never left the collector
    kinds = [(a["category"], a["event"]) for a in agent.get_alerts()]
    assert kinds == [("data_quality", "reading_rejected"), ("data_quality", "resolved")]


def test_router_down_agent_keeps_monitoring_and_keeps_alerts():
    router = FakeRouter(down=True)
    agent = build([mk(1, c=15000), mk(2, c=15000)], router)
    run_ticks(agent, 2)                                          # must not raise
    assert agent.router_connected is False and agent.forward_failures == 2
    alerts = agent.get_alerts()
    assert [a["event"] for a in alerts] == ["suspected", "confirmed"]   # rules still ran locally
    assert alerts[-1]["enrichment"]["status"] == "failed"               # but no evidence available
    assert agent.snapshot()["router"]["connected"] is False


def test_router_reports_downstream_error():
    router = FakeRouter(error_status=True)
    agent = build([mk(1)], router)
    run_ticks(agent, 1)
    assert agent.forward_failures == 1 and "analysis offline" in agent.last_error


def test_analysis_anomaly_alert_is_edge_triggered():
    flags = iter([True, True, False])
    router = FakeRouter(anomaly=lambda m: next(flags))
    agent = build([mk(1), mk(2), mk(3)], router)
    run_ticks(agent, 3)
    kinds = [(a["category"], a["event"]) for a in agent.get_alerts()]
    assert kinds == [("statistical_anomaly", "detected"), ("statistical_anomaly", "resolved")]


def test_stream_wrap_resets_monitors():
    router = FakeRouter()
    r = [mk(1, c=15000, cycle=0), mk(2, c=15000, cycle=0), mk(1, c=10000, cycle=1)]
    agent = build(r, router)
    run_ticks(agent, 3)
    assert agent.monitors["South"].leak_state == "none"          # fresh state in the new cycle
    assert agent.get_alerts()[-1]["event"] == "confirmed"        # nothing spurious after the wrap


def test_pause_resume_are_audited_and_stop_the_loop():
    async def go():
        router = FakeRouter()
        agent = build([mk(d) for d in range(1, 30)], router)
        await agent.start()
        await asyncio.sleep(0.15)
        assert agent.pause("tester") and not agent.pause("tester")
        await asyncio.sleep(0.05)
        frozen = agent.ticks
        await asyncio.sleep(0.15)
        assert agent.ticks == frozen                             # nothing happens while paused
        assert agent.resume("tester")
        await asyncio.sleep(0.15)
        assert agent.ticks > frozen
        await agent.stop()
        return agent
    agent = asyncio.run(go())
    ops = [(a["event"], a["observation"]["actor"]) for a in agent.get_alerts() if a["category"] == "operator_action"]
    assert ops == [("paused", "tester"), ("resumed", "tester")]


def test_alert_feed_filters():
    router = FakeRouter()
    agent = build([mk(1, c=15000), mk(2, c=15000)], router)
    run_ticks(agent, 2)
    assert len(agent.get_alerts(after_id=1)) == 1
    assert agent.get_alerts(zone="north") == []
    assert len(agent.get_alerts(zone="SOUTH")) == 2
    assert len(agent.get_alerts(limit=1)) == 1


def test_disabled_agent_does_not_start():
    async def go():
        agent = build([mk(1)], FakeRouter(), enabled=False)
        await agent.start()
        return agent
    agent = asyncio.run(go())
    assert agent.snapshot()["running"] is False and agent.ticks == 0


def test_weak_ir_matches_are_not_cited_as_evidence():
    from agent import _extract_docs
    ir = {"results": [
        {"document": "leak_detection_procedure.txt", "score": 0.3042, "snippet": "a"},
        {"document": "customer_complaint_handling.txt", "score": 0.1042, "snippet": "b"},
        {"document": "drought_response_plan.txt", "score": 0.0137, "snippet": "c"}]}
    assert [d["document"] for d in _extract_docs(ir, 0.15)] == ["leak_detection_procedure.txt"]
    assert _extract_docs({"results": []}, 0.15) == [] and _extract_docs(None, 0.15) == []
