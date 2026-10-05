"""
agent.py - the autonomous loop of the Collector Agent.

    SENSE  -> pull the next reading from the data stream, on its own schedule
    THINK  -> rules.py: validate it, apply the SOP leak / drought rules,
              decide whether it is normal, suspect, or must be quarantined
    ACT    -> forward valid readings to the Router (usage_reading), and when
              something is wrong, gather supporting context and raise an alert:
                * Router -> IR        policy_query   (which SOP applies?)
                * Router -> Analysis  forecast_request (drought: shortage risk)

Design rules
  * Hub-and-spoke: the collector only ever talks to the Router, using routes
    the Router already exposes. Nothing in the other agents has to change.
  * Graceful degradation: if the Router (or anything behind it) is down, the
    collector keeps monitoring locally and keeps its alert log.
  * Human in the loop: the agent only RECOMMENDS. It can be paused/resumed by
    an operator and every such command is written to the audit log.
  * Explainable: every alert names the rule, the source document, the
    observed numbers, and the caveats.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

import httpx

from rules import RuleConfig, ZoneMonitor, make_event

log = logging.getLogger("collector.agent")

AGENT_CONSUMER = "agent"   # this loop's private cursor on the data stream


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class AgentConfig:
    enabled: bool = True
    router_url: str = "http://localhost:8000"
    api_key: str = ""
    poll_seconds: float = 2.0            # normal sampling interval
    heightened_factor: float = 0.5       # sample faster while a zone is in alert
    min_poll_seconds: float = 0.05
    request_timeout: float = 5.0
    alert_log_size: int = 500
    seed_window: int = 10
    data_source: str = "replayed_historical_dataset"
    evidence_min_score: float = 0.15     # IR always returns its top-k; drop weak matches

    @classmethod
    def from_env(cls, api_key: str, data_source: str = "replayed_historical_dataset") -> "AgentConfig":
        def num(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, default))
            except (TypeError, ValueError):
                return default

        enabled = os.getenv("COLLECTOR_AGENT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
        return cls(
            enabled=enabled,
            router_url=os.getenv("ROUTER_URL", cls.router_url),
            api_key=api_key,
            poll_seconds=num("COLLECTOR_POLL_SECONDS", cls.poll_seconds),
            heightened_factor=num("COLLECTOR_HEIGHTENED_FACTOR", cls.heightened_factor),
            request_timeout=num("ROUTER_TIMEOUT", cls.request_timeout),
            evidence_min_score=num("EVIDENCE_MIN_SCORE", cls.evidence_min_score),
            data_source=data_source,
        )


def _summarize_analysis(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    analysis = (response or {}).get("analysis") or {}
    payload = analysis.get("payload") or {}
    if analysis.get("status") != "ok" or "is_anomaly" not in payload:
        return None
    return {
        "is_anomaly": payload.get("is_anomaly"),
        "z_score": payload.get("z_score"),
        "severity": payload.get("severity"),
        "baseline_mean": payload.get("baseline_mean"),
        "baseline_std": payload.get("baseline_std"),
    }


def _extract_docs(ir_payload: Optional[Dict[str, Any]], min_score: float = 0.0) -> List[Dict[str, Any]]:
    """Keep only genuinely relevant IR hits. The IR module returns its top-k
    documents even when the match is weak (e.g. a leak alert citing the drought
    plan at score 0.01), which would make an alert's "evidence" misleading.
    A hit must reach `min_score` AND at least 30% of the best hit's score."""
    results = [r for r in ((ir_payload or {}).get("results") or []) if isinstance(r.get("score"), (int, float))]
    if not results:
        return []
    top = max(r["score"] for r in results)
    cutoff = max(min_score, 0.3 * top)
    return [
        {"document": r.get("document"), "score": r.get("score"), "snippet": (r.get("snippet") or "")[:200]}
        for r in results if r["score"] >= cutoff
    ]


class CollectorAgent:
    def __init__(
        self,
        stream,
        seed_provider: Callable[[], Dict[str, List[Dict[str, Any]]]],
        rule_config: RuleConfig,
        config: AgentConfig,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.stream = stream
        self._seed_provider = seed_provider
        self.rule_config = rule_config
        self.cfg = config
        self._transport = transport          # injected by tests

        self.monitors: Dict[str, ZoneMonitor] = {}
        self.analysis_active: Dict[str, bool] = {}
        self.alerts: Deque[Dict[str, Any]] = deque(maxlen=config.alert_log_size)
        self._next_alert_id = 1

        self.paused = False
        self.mode = "normal"                 # "normal" | "heightened"
        self.ticks = 0
        self.forwarded = 0
        self.quarantined = 0
        self.forward_failures = 0
        self.router_connected: Optional[bool] = None
        self.last_error: Optional[str] = None
        self.cycle: Optional[int] = None
        self.last_sequence = 0
        self.last_reading_date: Optional[str] = None

        self._seed: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if not self.cfg.enabled:
            log.info("Collector agent loop is disabled (COLLECTOR_AGENT_ENABLED=false).")
            return
        if self._task and not self._task.done():
            return
        self._client = httpx.AsyncClient(timeout=self.cfg.request_timeout, transport=self._transport)
        self._task = asyncio.create_task(self._run(), name="collector-agent-loop")
        log.info("Collector agent started: polling every %.2fs, router at %s",
                 self.cfg.poll_seconds, self.cfg.router_url)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _run(self) -> None:
        while True:
            try:
                if not self.paused:
                    await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # never let the loop die
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("Collector agent tick failed")
            await asyncio.sleep(self._interval())

    def _interval(self) -> float:
        if self.mode == "heightened":
            return max(self.cfg.min_poll_seconds, self.cfg.poll_seconds * self.cfg.heightened_factor)
        return max(self.cfg.min_poll_seconds, self.cfg.poll_seconds)

    # ------------------------------------------------------------------ one cycle
    async def tick(self) -> Dict[str, Any]:
        """One sense -> think -> act cycle. Returns a small summary (used by tests)."""
        # SENSE
        reading = self.stream.next(AGENT_CONSUMER)
        zone = str(reading.get("zone") or reading.get("region") or "unknown")
        if self.cycle is not None and reading["cycle"] != self.cycle:
            self._start_new_cycle()
        self.cycle = reading["cycle"]
        self.last_sequence = reading["sequence_id"]
        self.last_reading_date = reading.get("date")
        self.ticks += 1

        # THINK
        assessment = self._monitor(zone).evaluate(reading)
        new_alerts = [self._record(ev, reading) for ev in assessment.events]

        # ACT
        response = None
        if assessment.forward:
            response = await self._forward(reading, zone, assessment)
        else:
            self.quarantined += 1

        if response is not None:
            analysis_event = self._analysis_event(zone, reading, response)
            if analysis_event:
                new_alerts.append(self._record(analysis_event, reading))

        for alert in new_alerts:
            await self._enrich(alert, reading, response)

        self._update_mode()
        return {"zone": zone, "quality": assessment.quality, "forwarded": response is not None,
                "alerts": [a["alert_id"] for a in new_alerts]}

    def _monitor(self, zone: str) -> ZoneMonitor:
        if self._seed is None:
            self._seed = self._seed_provider()
        if zone not in self.monitors:
            self.monitors[zone] = ZoneMonitor(zone, self.rule_config, self._seed.get(zone, []))
        return self.monitors[zone]

    def _start_new_cycle(self) -> None:
        """The replay wrapped around: dates repeat, so start every zone afresh."""
        log.info("Data stream wrapped to replay cycle %s; resetting zone monitors.", self.cycle)
        self.monitors.clear()
        self.analysis_active.clear()

    # ------------------------------------------------------------------ act: Router calls
    async def _post_router(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = f"{self.cfg.router_url.rstrip('/')}/router/process"
        try:
            resp = await self._client.post(url, json=message, headers={"X-API-Key": self.cfg.api_key})
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self._set_router_state(False, f"{type(exc).__name__}: {exc}")
            return None
        self._set_router_state(True)
        if data.get("status") != "ok":
            self.last_error = f"router reported: {data.get('error')}"
            return None
        return data

    def _set_router_state(self, connected: bool, error: Optional[str] = None) -> None:
        if connected and self.router_connected is False:
            log.info("Router connection restored.")
        if not connected and self.router_connected is not False:
            log.warning("Router unreachable (%s). Continuing to monitor locally.", error)
        self.router_connected = connected
        if error:
            self.last_error = error

    async def _forward(self, reading, zone: str, assessment) -> Optional[Dict[str, Any]]:
        message = {
            "agent": "collector",
            "zone": zone,
            "type": "usage_reading",
            "payload": {
                "date": reading.get("date"),
                "consumption_liters": float(reading["consumption_liters"]),
                "rainfall_mm": float(reading["rainfall_mm"]),
                "reservoir_pct_full": float(reading["reservoir_pct_full"]),
                "collector_assessment": {
                    "quality": assessment.quality,
                    "flags": assessment.flags,
                    "metrics": assessment.metrics,
                },
                "provenance": {
                    "source": self.cfg.data_source,
                    "sequence_id": reading.get("sequence_id"),
                    "cycle": reading.get("cycle"),
                },
            },
            "timestamp": now_iso(),
        }
        data = await self._post_router(message)
        if data is None:
            self.forward_failures += 1
        else:
            self.forwarded += 1
        return data

    async def _policy_query(self, zone: str, query: str) -> Optional[Dict[str, Any]]:
        return await self._post_router({
            "agent": "collector", "zone": zone, "type": "policy_query",
            "payload": {"query": query}, "timestamp": now_iso()})

    async def _forecast(self, zone: str, reading) -> Optional[Dict[str, Any]]:
        data = await self._post_router({
            "agent": "collector", "zone": zone, "type": "forecast_request",
            "payload": {"days_ahead": 7,
                        "rainfall_mm": float(reading["rainfall_mm"]),
                        "reservoir_pct_full": float(reading["reservoir_pct_full"])},
            "timestamp": now_iso()})
        analysis = (data or {}).get("analysis") or {}
        payload = analysis.get("payload") or {}
        if analysis.get("status") != "ok" or "shortage_risk" not in payload:
            return None
        return {k: payload.get(k) for k in
                ("days_ahead", "shortage_risk", "trend_direction", "forecast", "historical_p90")}

    # ------------------------------------------------------------------ alerts
    def _record(self, event: Dict[str, Any], reading: Dict[str, Any]) -> Dict[str, Any]:
        alert = dict(event)
        alert.update({
            "alert_id": self._next_alert_id,
            "timestamp": now_iso(),
            "sequence_id": reading.get("sequence_id"),
            "cycle": reading.get("cycle"),
            "data_source": self.cfg.data_source,
            "evidence": [],
            "analysis": None,
            "forecast": None,
            "enrichment": {"status": "not_required", "steps": []},
        })
        self._next_alert_id += 1
        self.alerts.append(alert)
        level = logging.WARNING if alert["severity"] != "info" else logging.INFO
        log.log(level, "ALERT #%d [%s] %s/%s zone=%s | %s", alert["alert_id"], alert["severity"].upper(),
                alert["category"], alert["event"], alert["zone"], alert["reason"])
        return alert

    def _analysis_event(self, zone: str, reading, response) -> Optional[Dict[str, Any]]:
        """Turn the Analysis agent's verdict into an (edge-triggered) alert."""
        verdict = _summarize_analysis(response)
        if verdict is None:
            return None
        anomalous = bool(verdict["is_anomaly"])
        active = self.analysis_active.get(zone, False)
        date = reading.get("date")
        source = "analysis-agent (z-score against trained baseline)"
        if anomalous and not active:
            self.analysis_active[zone] = True
            sev = {"critical": "critical", "high": "high"}.get(verdict["severity"], "warning")
            z = verdict["z_score"]
            return make_event(
                zone, "statistical_anomaly", "detected", sev, "AN-01", source,
                f"Analysis agent scored consumption of {float(reading['consumption_liters']):,.0f} L at "
                f"z = {z:+.2f} against this zone's trained baseline "
                f"(mean {verdict['baseline_mean']:,.0f} L, std {verdict['baseline_std']:,.0f} L).",
                f"Review recent flow data for {zone} and compare with the collector's own leak and drought findings.",
                {"z_score": z, "severity": verdict["severity"]}, date,
                ["The trained baseline covers the earlier part of the dataset; a rising seasonal trend can "
                 "push normal readings above it."])
        if not anomalous and active:
            self.analysis_active[zone] = False
            return make_event(
                zone, "statistical_anomaly", "resolved", "info", "AN-01", source,
                "Analysis agent now scores this zone's consumption as normal.", "No action needed.",
                {"z_score": verdict["z_score"]}, date)
        return None

    async def _enrich(self, alert: Dict[str, Any], reading, response) -> None:
        """Gather supporting context for an alert: the SOP text (IR) and, for
        drought escalations, a demand forecast (Analysis). Alerts already exist
        in the log before this runs, so a failure here never loses an alert."""
        if alert["severity"] == "info" or alert["category"] in ("data_quality", "operator_action"):
            return
        steps: List[Dict[str, Any]] = []
        alert["enrichment"] = {"status": "pending", "steps": steps}
        zone = alert["zone"]

        if response is not None:
            alert["analysis"] = _summarize_analysis(response)

        # Always use the collector's own evidence query. The Router attaches its
        # own documents to anomalous readings, but its query includes the
        # severity word ("high"), which matches unrelated billing text.
        data = await self._policy_query(zone, self._evidence_query(alert))
        docs = _extract_docs((data or {}).get("result"), self.cfg.evidence_min_score)
        # "ok" means the IR call succeeded; an empty list is a valid answer
        # ("no sufficiently relevant document"), not a failure.
        steps.append({"step": "policy_query", "ok": data is not None})
        alert["evidence"] = docs

        if alert["category"] == "drought" and alert["event"] == "escalated":
            forecast = await self._forecast(zone, reading)
            alert["forecast"] = forecast
            steps.append({"step": "forecast_request", "ok": forecast is not None})

        oks = [s["ok"] for s in steps]
        alert["enrichment"]["status"] = "complete" if all(oks) else ("partial" if any(oks) else "failed")

    @staticmethod
    def _evidence_query(alert: Dict[str, Any]) -> str:
        """Query text for the IR module. Zone names and severity words are left
        out on purpose: they are not in the documents and only add noise."""
        cat = alert["category"]
        if cat == "drought":
            return "drought response plan reservoir capacity stage restrictions"
        if cat == "leak":
            return "suspected leak flow exceeds rolling average consumption leak detection"
        return "abnormal flow exceeds rolling average leak detection"

    # ------------------------------------------------------------------ adaptive behaviour
    def _update_mode(self) -> None:
        """Sample faster while any zone is in an alert state."""
        alerting = any(m.is_alerting() for m in self.monitors.values()) or any(self.analysis_active.values())
        new_mode = "heightened" if alerting else "normal"
        if new_mode != self.mode:
            self.mode = new_mode
            log.info("Monitoring mode -> %s (poll interval now %.2fs)", new_mode, self._interval())

    # ------------------------------------------------------------------ operator controls (human oversight)
    def pause(self, actor: str = "operator") -> bool:
        if self.paused:
            return False
        self.paused = True
        self._operator_event("paused", actor)
        return True

    def resume(self, actor: str = "operator") -> bool:
        if not self.paused:
            return False
        self.paused = False
        self._operator_event("resumed", actor)
        return True

    def _operator_event(self, event: str, actor: str) -> None:
        self._record(make_event(
            "all", "operator_action", event, "info", "OPS-01", "operator command (audit log)",
            f"Autonomous monitoring {event} by {actor}.", "None - recorded for the audit trail.",
            {"actor": actor}, self.last_reading_date), {"sequence_id": self.last_sequence, "cycle": self.cycle})

    # ------------------------------------------------------------------ read-only views
    def snapshot(self) -> Dict[str, Any]:
        zones = {}
        for zone, m in sorted(self.monitors.items()):
            zones[zone] = {**m.state(), "analysis_anomaly_active": self.analysis_active.get(zone, False)}
        return {
            "agent": "collector",
            "enabled": self.cfg.enabled,
            "running": bool(self._task and not self._task.done()),
            "paused": self.paused,
            "mode": self.mode,
            "poll_interval_seconds": round(self._interval(), 3),
            "base_poll_interval_seconds": self.cfg.poll_seconds,
            "router": {"url": self.cfg.router_url, "connected": self.router_connected,
                       "last_error": self.last_error, "forward_failures": self.forward_failures},
            "counters": {"ticks": self.ticks, "forwarded": self.forwarded,
                         "quarantined": self.quarantined, "alerts_total": self._next_alert_id - 1},
            "stream": {"cycle": self.cycle, "last_sequence_id": self.last_sequence,
                       "last_reading_date": self.last_reading_date, "data_source": self.cfg.data_source},
            "zones": zones,
            "rules": asdict(self.rule_config),
        }

    @property
    def latest_alert_id(self) -> int:
        return self._next_alert_id - 1

    def get_alerts(self, limit: int = 50, zone: Optional[str] = None, after_id: int = 0) -> List[Dict[str, Any]]:
        items = [a for a in self.alerts
                 if a["alert_id"] > after_id and (zone is None or a["zone"].lower() == zone.lower())]
        return copy.deepcopy(items[-limit:])
