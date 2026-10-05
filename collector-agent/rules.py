"""
rules.py - Collector Agent decision rules (pure logic, no network / no I/O).

This is the "Think" step of the collector's sense -> think -> act loop.
Every rule is grounded in a document the IR module already holds, so each
alert can say *which rule fired and why*:

  LEAK-01  leak_detection_procedure.txt
           consumption > 140% of the 7-day rolling average with no rise in
           rainfall  -> suspected leak.  Repeated hits -> confirmed leak.
           Excess above the average >= 5,000 L/day -> "major" leak.
  DRT-01   drought_response_plan.txt
           reservoir < 30% / 20% / 10%  -> drought Stage 1 / 2 / 3.
  DQ-01..04 collector data-quality policy (this project's own rules)
           invalid readings are quarantined; implausible jumps and
           frozen sensors are flagged.

The rules are deterministic on purpose: for safety-relevant decisions the
group needs decisions that are explainable and repeatable in the viva.
Thresholds default to the SOP values and can be overridden via env vars.

The collector only ever RECOMMENDS an action. A human decides.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

LEAK_DOC = "leak_detection_procedure.txt"
DROUGHT_DOC = "drought_response_plan.txt"
DQ_POLICY = "collector data-quality policy"

SEVERITY_ORDER = {"info": 0, "warning": 1, "high": 2, "critical": 3}


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RuleConfig:
    # LEAK-01 (leak_detection_procedure.txt)
    leak_ratio: float = 1.40          # "more than 40 percent" above rolling average
    leak_window: int = 7              # "7-day rolling average"
    leak_major_liters: float = 5000.0  # minor < 5000 L/day <= major
    leak_confirm_hits: int = 2        # collector assumption: SOP does not define "confirmed"
    leak_confirm_window: int = 3
    # DRT-01 (drought_response_plan.txt): Stage 1 / 2 / 3 thresholds in % full
    drought_thresholds: tuple = (30.0, 20.0, 10.0)
    drought_hysteresis: float = 2.0   # points above a threshold needed to step DOWN a stage
    # Data quality
    reservoir_jump_pts: float = 25.0
    stuck_window: int = 10
    stuck_range: float = 0.5

    @classmethod
    def from_env(cls) -> "RuleConfig":
        return cls(
            leak_ratio=_env_float("LEAK_RATIO_THRESHOLD", cls.leak_ratio),
            leak_window=_env_int("LEAK_WINDOW", cls.leak_window),
            leak_major_liters=_env_float("LEAK_MAJOR_LITERS", cls.leak_major_liters),
            leak_confirm_hits=_env_int("LEAK_CONFIRM_HITS", cls.leak_confirm_hits),
            leak_confirm_window=_env_int("LEAK_CONFIRM_WINDOW", cls.leak_confirm_window),
            drought_hysteresis=_env_float("DROUGHT_HYSTERESIS_PCT", cls.drought_hysteresis),
            reservoir_jump_pts=_env_float("RESERVOIR_JUMP_PCT", cls.reservoir_jump_pts),
            stuck_window=_env_int("STUCK_SENSOR_WINDOW", cls.stuck_window),
            stuck_range=_env_float("STUCK_SENSOR_RANGE", cls.stuck_range),
        )


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
@dataclass
class Assessment:
    """Outcome of evaluating one reading."""

    quality: str                      # "ok" | "suspect" | "invalid"
    flags: List[str]
    forward: bool                     # False => quarantined, do not send downstream
    events: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def make_event(
    zone: str,
    category: str,
    event: str,
    severity: str,
    rule_id: str,
    source_document: str,
    reason: str,
    recommended_action: str,
    observation: Dict[str, Any],
    data_date: Optional[str],
    caveats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "zone": zone,
        "category": category,
        "event": event,
        "severity": severity,
        "rule_id": rule_id,
        "source_document": source_document,
        "reason": reason,
        "recommended_action": recommended_action,
        "observation": observation,
        "data_date": data_date,
        "caveats": caveats or [],
        # The agent never acts on its own: anything above "info" is a
        # recommendation for a human operator.
        "requires_human_review": SEVERITY_ORDER.get(severity, 0) >= 1,
    }


def _num(value: Any) -> Optional[float]:
    """Return a finite float, or None for missing / non-numeric / NaN / inf."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


FIELDS = ("consumption_liters", "rainfall_mm", "reservoir_pct_full")

DROUGHT_ACTIONS = {
    1: "Declare a Stage 1 drought alert and ask residents to voluntarily reduce outdoor water use.",
    2: "Declare Stage 2 mandatory restrictions: limit outdoor watering to two days per week.",
    3: "Declare Stage 3 emergency restrictions: ban non-essential outdoor use and require "
       "industrial users to cut consumption by 25 percent.",
}
DROUGHT_SEVERITY = {1: "warning", 2: "high", 3: "critical"}


# --------------------------------------------------------------------------
# Per-zone monitor: keeps memory between readings and decides
# --------------------------------------------------------------------------
class ZoneMonitor:
    """Holds one zone's recent history and evaluates each new reading.

    Alerts are *edge-triggered*: an alert is raised when a condition begins
    (or changes level) and a "resolved" record when it ends, instead of
    repeating the same alert on every reading while the condition persists.
    """

    def __init__(self, zone: str, cfg: RuleConfig, seed: Optional[List[Dict[str, Any]]] = None):
        self.zone = zone
        self.cfg = cfg
        self.reset(seed)

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: Optional[List[Dict[str, Any]]] = None) -> None:
        cfg = self.cfg
        self.consumption: Deque[float] = deque(maxlen=cfg.leak_window)
        self.rainfall: Deque[float] = deque(maxlen=cfg.leak_window)
        self.reservoir: Deque[float] = deque(maxlen=max(cfg.stuck_window, 2))
        self.leak_hits: Deque[bool] = deque(maxlen=cfg.leak_confirm_window)
        self.leak_state = "none"          # none | suspected | confirmed
        self.drought_stage = 0            # 0 = normal, 1..3 = SOP stages
        self.active = {"invalid": False, "stuck": False}
        self.last_date: Optional[str] = None
        # Warm-start the rolling windows from historical data so rules work
        # from the very first live reading. Seeding never raises alerts.
        for row in seed or []:
            c, r, v = (_num(row.get(k)) for k in FIELDS)
            if c is not None and r is not None and v is not None:
                self.consumption.append(c)
                self.rainfall.append(r)
                self.reservoir.append(v)
                self.last_date = row.get("date")

    def state(self) -> Dict[str, Any]:
        return {
            "last_date": self.last_date,
            "drought_stage": self.drought_stage,
            "leak_state": self.leak_state,
            "stuck_sensor_suspected": self.active["stuck"],
            "invalid_data": self.active["invalid"],
        }

    def is_alerting(self) -> bool:
        return self.leak_state != "none" or self.drought_stage > 0

    # -- validation --------------------------------------------------------
    def _validate(self, date: Any, vals: Dict[str, Optional[float]]) -> List[tuple]:
        problems = []
        if not isinstance(date, str) or not date.strip():
            problems.append(("missing_date", "date is missing"))
        for name in FIELDS:
            if vals[name] is None:
                problems.append((f"missing_{name}", f"{name} is missing or not a finite number"))
        c, r, v = (vals[k] for k in FIELDS)
        if c is not None and c <= 0:
            problems.append(("non_positive_consumption", f"consumption {c:g} L must be positive"))
        if r is not None and r < 0:
            problems.append(("negative_rainfall", f"rainfall {r:g} mm cannot be negative"))
        if v is not None and not (0 <= v <= 100):
            problems.append(("reservoir_out_of_range", f"reservoir level {v:g}% is outside 0-100"))
        if isinstance(date, str) and self.last_date is not None and date <= self.last_date:
            problems.append(("stale_or_duplicate_date",
                             f"date {date} is not after the last accepted date {self.last_date}"))
        return problems

    # -- main entry point --------------------------------------------------
    def evaluate(self, reading: Dict[str, Any]) -> Assessment:
        cfg, zone = self.cfg, self.zone
        date = reading.get("date")
        vals = {k: _num(reading.get(k)) for k in FIELDS}

        # 1. Hard validity: invalid data is quarantined, never analysed.
        problems = self._validate(date, vals)
        if problems:
            events = []
            if not self.active["invalid"]:
                self.active["invalid"] = True
                events.append(make_event(
                    zone, "data_quality", "reading_rejected", "warning", "DQ-01", DQ_POLICY,
                    "Reading rejected and NOT forwarded to analysis: "
                    + "; ".join(msg for _, msg in problems) + ".",
                    f"Check the {zone} data feed / sensor. Further invalid readings are "
                    "quarantined silently until valid data returns.",
                    {"date": date, "problems": [code for code, _ in problems]}, date))
            return Assessment("invalid", [code for code, _ in problems], False, events)

        events: List[Dict[str, Any]] = []
        if self.active["invalid"]:
            self.active["invalid"] = False
            events.append(make_event(
                zone, "data_quality", "resolved", "info", "DQ-01", DQ_POLICY,
                "Valid readings have resumed.", "No action needed.", {"date": date}, date))

        c, rain, res = vals["consumption_liters"], vals["rainfall_mm"], vals["reservoir_pct_full"]
        flags: List[str] = []
        metrics: Dict[str, Any] = {}

        # 2. Soft data-quality flag: implausible one-step reservoir change (DQ-03).
        prev_res = self.reservoir[-1] if self.reservoir else None
        if prev_res is not None and abs(res - prev_res) > cfg.reservoir_jump_pts:
            flags.append("reservoir_jump")
            events.append(make_event(
                zone, "data_quality", "implausible_jump", "warning", "DQ-03", DQ_POLICY,
                f"Reservoir level changed by {abs(res - prev_res):.1f} points in one reading "
                f"({prev_res:.1f}% -> {res:.1f}%), above the {cfg.reservoir_jump_pts:g}-point "
                "plausibility limit.",
                "Verify this reading against the reservoir sensor before acting on drought staging.",
                {"previous_pct": prev_res, "current_pct": res}, date))

        # 3. LEAK-01 - uses the window BEFORE this reading is absorbed.
        leak_events, leak_hit = self._leak_rule(c, rain, date, metrics)
        events += leak_events

        # Absorb the reading into memory. Leak-suspect consumption is kept OUT
        # of the rolling baseline: otherwise a sustained leak inflates its own
        # 7-day average and hides itself from the rule.
        if not leak_hit:
            self.consumption.append(c)
        self.rainfall.append(rain)
        self.reservoir.append(res)
        self.last_date = date

        # 4. DQ-04 stuck / frozen sensor.
        events += self._stuck_rule(res, date, flags)

        # 5. DRT-01 drought staging.
        events += self._drought_rule(res, date, metrics)

        quality = "suspect" if flags else "ok"
        return Assessment(quality, flags, True, events, metrics)

    # -- rules ---------------------------------------------------------------
    def _leak_rule(self, c: float, rain: float, date: str, metrics: Dict[str, Any]):
        """Returns (events, hit). `hit` readings are excluded from the baseline.

        Known limitation: because leak-suspect readings never enter the
        baseline, a genuine permanent step-up in demand (e.g. a new industrial
        user) keeps the zone in "confirmed" until an operator resets it.
        """
        cfg, zone = self.cfg, self.zone
        if len(self.consumption) < cfg.leak_window:
            return [], False
        avg = _mean(self.consumption)
        rain_avg = _mean(self.rainfall)
        if avg <= 0:
            return [], False

        ratio = c / avg
        excess = c - avg
        rain_rose = rain > rain_avg                     # "no corresponding increase in rainfall"
        hit = ratio > cfg.leak_ratio and not rain_rose
        metrics.update({"leak_ratio": round(ratio, 3), "rolling_avg_liters": round(avg, 1),
                        "excess_liters": round(excess, 1)})

        self.leak_hits.append(hit)
        hits = sum(self.leak_hits)
        major = excess > cfg.leak_major_liters
        pct = (ratio - 1) * 100
        obs = {"consumption_liters": c, "rolling_avg_liters": round(avg, 1),
               "ratio": round(ratio, 3), "excess_liters": round(excess, 1),
               "rainfall_mm": rain, "rainfall_avg_mm": round(rain_avg, 2)}
        events: List[Dict[str, Any]] = []
        maintenance_caveat = ("Scheduled-maintenance status is not available to the collector; "
                              "verify before dispatching a crew.")

        if hit:
            if self.leak_state == "none":
                self.leak_state = "suspected"
                events.append(make_event(
                    zone, "leak", "suspected", "high" if major else "warning", "LEAK-01", LEAK_DOC,
                    f"Consumption of {c:,.0f} L is {pct:.0f}% above the {cfg.leak_window}-day average "
                    f"({avg:,.0f} L) with no rise in rainfall ({rain:.1f} mm vs {rain_avg:.1f} mm recent average).",
                    "Monitor this zone closely. If the pattern repeats it is escalated to a confirmed leak "
                    "(SOP: dispatch a field technician within 24 hours of a confirmed anomaly).",
                    obs, date, [maintenance_caveat]))
            if self.leak_state == "suspected" and hits >= cfg.leak_confirm_hits:
                self.leak_state = "confirmed"
                events.append(make_event(
                    zone, "leak", "confirmed", "critical" if major else "high", "LEAK-01", LEAK_DOC,
                    f"Leak pattern seen on {hits} of the last {len(self.leak_hits)} readings; latest consumption "
                    f"{c:,.0f} L is {pct:.0f}% above the {cfg.leak_window}-day average ({avg:,.0f} L). "
                    f"Estimated excess about {excess:,.0f} L/day "
                    f"({'MAJOR' if major else 'minor'} against the SOP threshold of {cfg.leak_major_liters:,.0f} L/day).",
                    ("Immediately isolate the affected pipe segment and carry out emergency repair within 24 hours."
                     if major else
                     "Dispatch a field technician within 24 hours; log the leak and schedule routine repair "
                     "within one week."),
                    obs, date,
                    [maintenance_caveat,
                     f"'Confirmed' means the rule fired on {cfg.leak_confirm_hits} of the last "
                     f"{cfg.leak_confirm_window} readings - a collector assumption, the SOP does not define it."]))
        elif self.leak_state != "none" and hits == 0:
            previous = self.leak_state
            self.leak_state = "none"
            events.append(make_event(
                zone, "leak", "resolved", "info", "LEAK-01", LEAK_DOC,
                f"No leak pattern in the last {len(self.leak_hits)} readings; consumption is back within "
                f"normal range (previous state: {previous}).",
                "Close the incident once field verification is complete.", obs, date))
        return events, hit

    def _stuck_rule(self, res: float, date: str, flags: List[str]) -> List[Dict[str, Any]]:
        cfg, zone = self.cfg, self.zone
        if len(self.reservoir) < cfg.stuck_window:
            return []
        window = list(self.reservoir)[-cfg.stuck_window:]
        rng = max(window) - min(window)
        stuck = rng <= cfg.stuck_range
        events: List[Dict[str, Any]] = []
        if stuck:
            flags.append("possible_stuck_sensor")
            if not self.active["stuck"]:
                self.active["stuck"] = True
                events.append(make_event(
                    zone, "data_quality", "suspected_stuck_sensor", "warning", "DQ-04", DQ_POLICY,
                    f"Reservoir level has varied by only {rng:.2f} points over the last {cfg.stuck_window} "
                    f"readings (about {res:.1f}%) - possible stuck or frozen sensor.",
                    f"Inspect the {zone} reservoir level sensor / telemetry link. Do not rely on this "
                    "zone's reservoir level for drought staging until verified.",
                    {"range_pts": round(rng, 3), "window": cfg.stuck_window, "level_pct": res}, date))
        elif self.active["stuck"]:
            self.active["stuck"] = False
            events.append(make_event(
                zone, "data_quality", "resolved", "info", "DQ-04", DQ_POLICY,
                f"Reservoir level is varying again (range {rng:.2f} points).", "No action needed.",
                {"range_pts": round(rng, 3)}, date))
        return events

    def _next_drought_stage(self, level: float) -> int:
        t1, t2, t3 = self.cfg.drought_thresholds
        raw = 3 if level < t3 else 2 if level < t2 else 1 if level < t1 else 0
        current = self.drought_stage
        if raw >= current:
            return raw                                   # escalate (or hold) immediately
        # Stepping DOWN needs to clear the current stage's threshold by a margin,
        # so a level hovering around a threshold does not flap between stages.
        current_threshold = self.cfg.drought_thresholds[current - 1]
        return raw if level >= current_threshold + self.cfg.drought_hysteresis else current

    def _drought_rule(self, res: float, date: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        cfg, zone = self.cfg, self.zone
        old = self.drought_stage
        new = self._next_drought_stage(res)
        metrics["drought_stage"] = new
        if new == old:
            return []
        self.drought_stage = new
        obs = {"reservoir_pct_full": res, "previous_stage": old, "new_stage": new}
        caveats = ["Stage is evaluated on every reading; the drought plan calls for weekly operations review.",
                   "A single low reading can be a sensor error - compare with neighbouring readings."]
        if new > old:
            thr = cfg.drought_thresholds[new - 1]
            return [make_event(
                zone, "drought", "escalated", DROUGHT_SEVERITY[new], "DRT-01", DROUGHT_DOC,
                f"Reservoir at {res:.1f}% is below the Stage {new} threshold of {thr:g}% "
                f"(zone moved from Stage {old} to Stage {new}).",
                DROUGHT_ACTIONS[new], obs, date, caveats)]
        thr_old = cfg.drought_thresholds[old - 1]
        if new == 0:
            return [make_event(
                zone, "drought", "cleared", "info", "DRT-01", DROUGHT_DOC,
                f"Reservoir recovered to {res:.1f}% (above the {thr_old:g}% Stage {old} threshold plus a "
                f"{cfg.drought_hysteresis:g}-point margin). Drought stage cleared.",
                "Operations team to review whether restrictions can be lifted.", obs, date, caveats)]
        return [make_event(
            zone, "drought", "de-escalated", "info", "DRT-01", DROUGHT_DOC,
            f"Reservoir recovered to {res:.1f}%; zone moved from Stage {old} down to Stage {new}.",
            "Operations team to review whether restrictions can be relaxed.", obs, date, caveats)]
