"""Unit tests for the collector's decision rules (rules.py). No network needed."""
from rules import RuleConfig, ZoneMonitor


def seed(n=7, consumption=10000.0, rain=0.0, reservoir=60.0, start_day=1):
    return [
        {"date": f"2023-05-{start_day + i:02d}", "consumption_liters": consumption,
         "rainfall_mm": rain, "reservoir_pct_full": reservoir + i * 0.7}   # slight drift: not "stuck"
        for i in range(n)
    ]


def reading(day, consumption=10000.0, rain=0.0, reservoir=60.0):
    return {"date": f"2023-06-{day:02d}", "consumption_liters": consumption,
            "rainfall_mm": rain, "reservoir_pct_full": reservoir}


def monitor(**cfg):
    return ZoneMonitor("South", RuleConfig(**cfg), seed())


def kinds(assessment):
    return [(e["category"], e["event"]) for e in assessment.events]


# ---------------------------------------------------------------- leak rule
def test_normal_reading_raises_nothing_and_is_forwarded():
    a = monitor().evaluate(reading(1))
    assert a.quality == "ok" and a.forward and a.events == []


def test_leak_rule_matches_sop_40_percent_threshold():
    m = monitor()
    a = m.evaluate(reading(1, consumption=14000))       # exactly +40%: SOP says "MORE than 40"
    assert kinds(a) == []
    m2 = monitor()
    a2 = m2.evaluate(reading(1, consumption=14100))     # +41%
    assert kinds(a2) == [("leak", "suspected")]
    assert a2.events[0]["source_document"] == "leak_detection_procedure.txt"
    assert a2.events[0]["rule_id"] == "LEAK-01"


def test_rainfall_increase_suppresses_leak():
    a = monitor().evaluate(reading(1, consumption=15000, rain=12.0))
    assert kinds(a) == []                                # high use explained by rain


def test_leak_is_edge_triggered_and_escalates_to_confirmed():
    m = monitor()
    assert kinds(m.evaluate(reading(1, consumption=15000))) == [("leak", "suspected")]
    assert kinds(m.evaluate(reading(2, consumption=15000))) == [("leak", "confirmed")]
    # persisting leak must NOT re-alert on every reading
    assert kinds(m.evaluate(reading(3, consumption=15000))) == []
    assert m.leak_state == "confirmed"


def test_minor_vs_major_leak_uses_5000_litre_threshold():
    minor = monitor()
    minor.evaluate(reading(1, consumption=14500))
    ev = minor.evaluate(reading(2, consumption=14500)).events[0]
    assert ev["severity"] == "high" and "minor" in ev["reason"]

    major = monitor()
    major.evaluate(reading(1, consumption=16000))
    ev = major.evaluate(reading(2, consumption=16000)).events[0]
    assert ev["severity"] == "critical" and "MAJOR" in ev["reason"]
    assert "isolate" in ev["recommended_action"].lower()


def test_leak_resolves_after_quiet_readings():
    m = monitor()
    m.evaluate(reading(1, consumption=15000))
    out = []
    for d in range(2, 6):
        out += kinds(m.evaluate(reading(d, consumption=10000)))
    assert ("leak", "resolved") in out and m.leak_state == "none"


# ------------------------------------------------------------ drought rule
def low_monitor():
    """Zone whose reservoir history is already near the drought thresholds."""
    return ZoneMonitor("West", RuleConfig(), seed(reservoir=32))


def test_drought_stages_follow_sop_thresholds():
    m = low_monitor()
    assert kinds(m.evaluate(reading(1, reservoir=35))) == []
    e = m.evaluate(reading(2, reservoir=29)).events[0]
    assert (e["event"], e["severity"]) == ("escalated", "warning") and m.drought_stage == 1
    m.evaluate(reading(3, reservoir=19)); assert m.drought_stage == 2
    e = m.evaluate(reading(4, reservoir=9)).events[0]
    assert e["severity"] == "critical" and m.drought_stage == 3
    assert "25 percent" in e["recommended_action"]


def test_drought_alert_only_on_stage_change():
    m = low_monitor()
    m.evaluate(reading(1, reservoir=25))
    assert kinds(m.evaluate(reading(2, reservoir=24))) == []      # still Stage 1 -> silent


def test_drought_hysteresis_prevents_flapping():
    m = low_monitor()
    m.evaluate(reading(1, reservoir=28))                          # Stage 1
    assert kinds(m.evaluate(reading(2, reservoir=30.5))) == []    # above 30 but inside margin
    assert m.drought_stage == 1
    assert kinds(m.evaluate(reading(3, reservoir=33))) == [("drought", "cleared")]


# ------------------------------------------------------------ data quality
def test_invalid_reading_is_quarantined_once_then_resolved():
    m = monitor()
    a = m.evaluate(reading(1, consumption=-5))
    assert a.quality == "invalid" and not a.forward
    assert kinds(a) == [("data_quality", "reading_rejected")]
    assert kinds(m.evaluate(reading(2, consumption=-5))) == []    # not repeated
    assert kinds(m.evaluate(reading(3))) == [("data_quality", "resolved")]


def test_nan_missing_and_out_of_range_are_rejected():
    for bad in [dict(consumption=float("nan")), dict(consumption=None), dict(reservoir=101), dict(rain=-1)]:
        assert not monitor().evaluate(reading(1, **bad)).forward


def test_duplicate_or_out_of_order_date_is_rejected():
    m = monitor()
    assert m.evaluate(reading(5)).forward
    a = m.evaluate(reading(5))
    assert not a.forward and "stale_or_duplicate_date" in a.flags


def test_reservoir_jump_is_flagged_but_still_forwarded():
    m = monitor()
    m.evaluate(reading(1, reservoir=70))
    a = m.evaluate(reading(2, reservoir=30))
    assert a.forward and a.quality == "suspect" and "reservoir_jump" in a.flags


def test_stuck_sensor_detected_then_cleared():
    m = ZoneMonitor("Central", RuleConfig(), seed())
    out = []
    for d in range(1, 12):
        out += kinds(m.evaluate(reading(d, reservoir=42.0)))
    assert ("data_quality", "suspected_stuck_sensor") in out
    assert out.count(("data_quality", "suspected_stuck_sensor")) == 1   # edge-triggered
    assert ("data_quality", "resolved") in kinds(m.evaluate(reading(12, reservoir=47.0)))


def test_reset_clears_state_but_keeps_seed_warm():
    m = monitor()
    m.evaluate(reading(1, consumption=15000)); m.evaluate(reading(2, reservoir=15))
    m.reset(seed())
    assert m.leak_state == "none" and m.drought_stage == 0 and len(m.consumption) == 7
    assert m.evaluate(reading(1)).forward                        # date reuse OK after reset


def test_sustained_leak_cannot_hide_by_inflating_its_own_baseline():
    """Regression: leak readings must stay out of the 7-day baseline."""
    m = monitor()
    m.evaluate(reading(1, consumption=14500))
    assert list(m.consumption) == [10000.0] * 7          # leak reading NOT absorbed
    a = m.evaluate(reading(2, consumption=14500))        # still measured against 10,000
    assert kinds(a) == [("leak", "confirmed")]
