"""Tests for the rule-based named-entity recogniser (ner.py). No network."""
import json
import os
import time

import pytest

from ner import entity_key, extract_entities, matching_entities

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def keys(text):
    return [entity_key(e) for e in extract_entities(text)]


# ------------------------------------------------------------ each entity type, normalised
@pytest.mark.parametrize("text, expected", [
    ("Zone 4 has a leak", [("ZONE", "4")]),
    ("the west region", [("ZONE", "West")]),
    ("Zone Four has low pressure", [("ZONE", "4")]),
    ("area 7 is dry", [("ZONE", "7")]),
    ("declare a Stage 1 drought alert", [("DROUGHT_STAGE", "1")]),
    ("we have entered Stage III restrictions", [("DROUGHT_STAGE", "3")]),
    ("stage two applies", [("DROUGHT_STAGE", "2")]),
    ("below 30 percent capacity", [("PERCENTAGE", "30")]),
    ("up 42% on last week", [("PERCENTAGE", "42")]),
    ("the reservoir is at twenty percent", [("FACILITY", "reservoir"), ("PERCENTAGE", "20")]),
    ("under 5000 liters per day", [("VOLUME", "5000 liters")]),
    ("consumption of 11,398 L", [("VOLUME", "11398 liters")]),
    ("loses 5,000 litres/day", [("VOLUME", "5000 liters")]),
    ("stays below 15 psi", [("PRESSURE", "15 psi")]),
    ("within 24 hours", [("DURATION", "24 hour")]),
    ("the 7-day rolling average", [("DURATION", "7 day")]),
    ("a six month average", [("DURATION", "6 month")]),
    ("resolved within 5 business days", [("DURATION", "5 day")]),
    ("for at least one minute", [("DURATION", "1 minute")]),
    ("it took 1.5 hours", [("DURATION", "1.5 hour")]),
    ("between 0.2 and 4.0 milligrams per liter", [("CONCENTRATION", "0.2-4 mg/l")]),
    ("chlorine at 0.5 mg per liter", [("WATER_QUALITY", "chlorine"), ("CONCENTRATION", "0.5 mg/l")]),
    ("tested for chlorine residual", [("WATER_QUALITY", "chlorine_residual")]),
    ("turbidity spikes and bacterial presence", [("WATER_QUALITY", "turbidity"), ("WATER_QUALITY", "bacteria")]),
    ("the pumping stations and hydrants", [("FACILITY", "pump_station"), ("FACILITY", "hydrant")]),
    ("trunk mains and isolation valves", [("FACILITY", "trunk_main"), ("FACILITY", "valve")]),
    ("the treatment facility", [("FACILITY", "treatment_plant")]),
])
def test_entities_are_found_and_normalised(text, expected):
    assert keys(text) == expected


def test_typed_fields_are_usable_by_other_agents():
    vol = extract_entities("under 5000 liters per day")[0]
    assert (vol["value"], vol["unit"], vol["per"]) == (5000, "liters", "day")
    dur = extract_entities("within 24 hours")[0]
    assert (dur["value"], dur["unit"], dur["hours"]) == (24, "hour", 24)
    assert extract_entities("two days")[0]["hours"] == 48
    rng = extract_entities("between 0.2 and 4.0 milligrams per liter")[0]
    assert (rng["min"], rng["max"], rng["unit"]) == (0.2, 4, "mg/l")
    assert extract_entities("Stage 3")[0]["value"] == 3           # a number, not a string


# ------------------------------------------------------------ structure of the output
def test_offsets_are_exact_and_entities_never_overlap():
    text = ("When reservoir levels fall below 30 percent, declare Stage 1. Zone 4 hours ago, "
            "between 0.2 and 4.0 milligrams per liter, 5,000 liters per day within 24 hours in the west.")
    ents = extract_entities(text)
    assert ents and all(text[e["start"]:e["end"]] == e["text"] for e in ents)
    assert all(a["end"] <= b["start"] for a, b in zip(ents, ents[1:]))


def test_longest_match_wins_over_parts():
    assert keys("between 0.2 and 4.0 milligrams per liter") == [("CONCENTRATION", "0.2-4 mg/l")]   # not two singles
    assert keys("chlorine residual") == [("WATER_QUALITY", "chlorine_residual")]                    # not "chlorine"
    assert keys("zone 4 hours ago") == [("ZONE", "4")]                                              # "4 hours" loses to "zone 4"


# ------------------------------------------------------------ what must NOT be extracted
@pytest.mark.parametrize("text", [
    "I live at 45 Palm Road, flat 12",
    "the pipe is 10 meters long", "we need a 5 meter pipe", "a 2-meter section",
    "the second stage of repairs starts next week",
    "Central Park is 5 km away", "Central Station is closed today",
    "call 1800 555 0199 between 9 and 5", "open from 8 am to 6 pm",
    "residents in the eastern district", "it happens twice weekly", "thank you for your help", "",
])
def test_traps_produce_no_entities(text):
    assert extract_entities(text) == []


def test_stage_i_needs_a_capital_so_the_pronoun_is_not_a_stage():
    assert keys("Stage I alert declared") == [("DROUGHT_STAGE", "1")]
    assert keys("at this stage i think it is fine") == []


def test_known_limitation_zone_names_are_ambiguous_outside_the_domain():
    """Documented weakness: a bare zone name cannot be told apart from a place name."""
    assert keys("the South Pole is very cold") == [("ZONE", "South")]


def test_meter_as_equipment_is_still_found():
    assert keys("my meter is broken") == [("FACILITY", "meter")]
    assert keys("flow meters must be calibrated") == [("FACILITY", "meter")]


# ------------------------------------------------------------ robustness
def test_none_and_empty_are_safe():
    assert extract_entities("") == [] and extract_entities(None) == []


@pytest.mark.parametrize("text", ["1" * 200000, "1," * 100000, "one " * 50000, "stage " * 40000, " " * 200000, "7-" * 100000], ids=["digits", "commas", "words", "stages", "spaces", "hyphens"])
def test_hostile_input_cannot_make_it_slow(text):
    t = time.time()
    extract_entities(text)
    assert time.time() - t < 2.0


# ------------------------------------------------------------ comparing a question with a passage
def test_matching_entities_uses_type_and_normalised_value():
    q = extract_entities("Stage 3 restrictions within 24 hours in zone 2")
    p = extract_entities("Stage 3 emergency restrictions apply. Repair within one day or 24 hours.")
    got = {(m["type"], m["normalized"]) for m in matching_entities(q, p)}
    assert got == {("DROUGHT_STAGE", "3"), ("DURATION", "24 hour")}        # zone 2 is not in the passage
    assert matching_entities(q, []) == [] and matching_entities([], p) == []


def test_matching_entities_does_not_repeat():
    q = extract_entities("stage 3 and again stage 3")
    p = extract_entities("Stage 3 applies")
    assert len(matching_entities(q, p)) == 1


# ------------------------------------------------------------ regression guard on the annotated sets
def _f1(path, section):
    data = json.load(open(os.path.join(ROOT, "eval", path), encoding="utf-8"))
    rows = [(q["text"], {tuple(e) for e in q["gold"]}) for q in data[section]] if section == "queries" else None
    tp = fp = fn = 0
    for text, gold in rows:
        pred = {entity_key(e) for e in extract_entities(text)}
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def test_development_questions_stay_perfect():
    assert _f1("ner_gold.json", "queries") == 1.0


def test_fresh_set_does_not_regress():
    assert _f1("ner_fresh.json", "queries") >= 0.95


def test_documents_gold_stays_perfect():
    gold = json.load(open(os.path.join(ROOT, "eval", "ner_gold.json"), encoding="utf-8"))["documents"]
    for name, expected in gold.items():
        text = open(os.path.join(ROOT, "data", "documents", name), encoding="utf-8").read()
        assert {entity_key(e) for e in extract_entities(text)} == {tuple(e) for e in expected}, name
