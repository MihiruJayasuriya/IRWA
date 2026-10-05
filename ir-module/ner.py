"""
ner.py - rule-based named-entity recognition (NER) for water-utility text.

It finds the structured facts inside free text, in citizen questions and in the policy
passages that answer them, and normalises them so other agents can compare and
compute with them instead of re-parsing strings:

  ZONE           "Zone 4", "west"                     -> value "4" / "West"
  DROUGHT_STAGE  "Stage 3"                             -> 3
  PERCENTAGE     "20 percent", "42%"                   -> 20 / 42
  VOLUME         "5,000 liters per day", "11,398 L"    -> 5000 liters (per day)
  PRESSURE       "15 psi"                              -> 15 psi
  DURATION       "24 hours", "7-day", "six month"      -> 24 hour, 7 day, 6 month (+ hours)
  CONCENTRATION  "between 0.2 and 4.0 milligrams per liter" -> 0.2-4 mg/l
  WATER_QUALITY  "chlorine residual", "turbidity"      -> chlorine_residual, turbidity
  FACILITY       "reservoir", "pumping station"        -> reservoir, pump_station

Method: hand-written patterns (regular expressions) and small gazetteers, then a
longest-match-wins pass so entities never overlap. No model, no training data: every
decision can be read, explained and challenged. That is a deliberate trade-off. It is
precise on the entity kinds above and blind to anything not listed (clock times,
frequencies such as "twice weekly", dates).

Each entity: type, text (as written), value (typed), normalized (canonical string used for
matching), start/end (character offsets), and unit / per / hours / min / max where relevant.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Tuple

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
}

_NUM = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"           # 5,000  8011  0.2
_WORDNUM = "(?:" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + ")"
_NUMW = f"(?:{_NUM}|{_WORDNUM})"
_START = r"(?<![\w.,])"                                            # never start in the middle of a number/word
_FLAGS = re.IGNORECASE


def _to_number(token: str) -> float:
    token = token.lower()
    return float(_WORD_NUMBERS[token]) if token in _WORD_NUMBERS else float(token.replace(",", ""))


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _num(x: float):
    return int(x) if float(x).is_integer() else x


# --------------------------------------------------------------------------- builders
# Each builder receives the regex match and returns the entity fields (except start/end/text).
def _zone_ref(m):
    v = m.group("v")
    if v.isdigit():
        pass
    elif v.lower() in _WORD_NUMBERS:              # "Zone Four" -> "4"
        v = str(_WORD_NUMBERS[v.lower()])
    else:
        v = v.title()
    return {"type": "ZONE", "value": v, "normalized": v}


_ROMAN = {"i": 1, "ii": 2, "iii": 3}


def _stage(m):
    tok = m.group("v").lower()
    n = int(tok) if tok.isdigit() else (_WORD_NUMBERS.get(tok) or _ROMAN[tok])
    return {"type": "DROUGHT_STAGE", "value": n, "normalized": str(n)}


def _percent(m):
    n = _to_number(m.group("n"))
    return {"type": "PERCENTAGE", "value": _num(n), "normalized": _fmt(n)}


_VOLUME_UNITS = {"l": "liters", "liter": "liters", "liters": "liters", "litre": "liters", "litres": "liters",
                 "kl": "kiloliters", "kiloliter": "kiloliters", "kiloliters": "kiloliters",
                 "kilolitre": "kiloliters", "kilolitres": "kiloliters",
                 "m3": "m3", "m³": "m3", "gallon": "gallons", "gallons": "gallons", "mld": "mld"}


def _volume(m):
    n = _to_number(m.group("n"))
    raw = re.sub(r"\s+", " ", m.group("u").lower())
    unit = "m3" if raw.startswith("cubic") else _VOLUME_UNITS[raw]
    out = {"type": "VOLUME", "value": _num(n), "unit": unit, "normalized": f"{_fmt(n)} {unit}"}
    if m.group("per"):
        out["per"] = m.group("per").lower()
    return out


def _pressure(m):
    n = _to_number(m.group("n"))
    unit = m.group("u").lower()
    return {"type": "PRESSURE", "value": _num(n), "unit": unit, "normalized": f"{_fmt(n)} {unit}"}


def _conc_range(m):
    a, b = _to_number(m.group("a")), _to_number(m.group("b"))
    return {"type": "CONCENTRATION", "min": _num(a), "max": _num(b), "unit": "mg/l",
            "value": [_num(a), _num(b)], "normalized": f"{_fmt(a)}-{_fmt(b)} mg/l"}


def _conc_single(m):
    n = _to_number(m.group("n"))
    return {"type": "CONCENTRATION", "value": _num(n), "unit": "mg/l", "normalized": f"{_fmt(n)} mg/l"}


_HOURS_PER = {"minute": 1 / 60, "hour": 1, "day": 24, "week": 168}


def _duration(m):
    n = _to_number(m.group("n"))
    raw = m.group("u").lower()
    unit = {"min": "minute", "mins": "minute", "hr": "hour", "hrs": "hour"}.get(raw, raw.rstrip("s"))
    out = {"type": "DURATION", "value": _num(n), "unit": unit, "normalized": f"{_fmt(n)} {unit}"}
    if unit in _HOURS_PER:
        out["hours"] = round(n * _HOURS_PER[unit], 4)
    return out


def _gazetteer(kind: str, canonical: str) -> Callable:
    return lambda m: {"type": kind, "value": canonical, "normalized": canonical}


# --------------------------------------------------------------------------- pattern table
_PATTERNS: List[Tuple[re.Pattern, Callable]] = [
    (re.compile(rf"{_START}\b(?:zone|area|region)\s+(?P<v>\d+|one|two|three|four|five|six|seven|eight|nine|ten|north|south|east|west|central)\b", _FLAGS), _zone_ref),
    # Zone names are ambiguous outside the domain: "Central Park" is a place, not a zone.
    (re.compile(r"\b(?P<v>north|south|east|west|central)\b(?!\s+(?:park|street|road|avenue|city|station|square|hospital|school|market|mall))", _FLAGS), _zone_ref),
    (re.compile(rf"\bstage\s+(?P<v>\d+|one|two|three|iii|ii|(?-i:I))\b", _FLAGS), _stage),
    (re.compile(rf"{_START}(?P<n>{_NUMW})\s*(?:%|(?:percent|per\s?cent)\b)", _FLAGS), _percent),
    (re.compile(rf"{_START}(?P<n>{_NUM})\s*(?P<u>liters?|litres?|kiloliters?|kilolitres?|kl|m3|m³|"
                rf"cubic\s+met(?:er|re)s?|gallons?|mld|l)\b(?:\s*(?:/|per|a)\s*(?P<per>day|hour|week|month|person)\b)?", _FLAGS), _volume),
    (re.compile(rf"{_START}(?P<n>{_NUM})\s*(?P<u>psi|kpa|mbar|bar)\b", _FLAGS), _pressure),
    (re.compile(rf"{_START}(?:between\s+)?(?P<a>{_NUM})\s*(?:and|to|-|–)\s*(?P<b>{_NUM})\s*"
                rf"(?:mg\s?/\s?l|mg\s+per\s+lit(?:er|re)s?|milligrams?\s+per\s+lit(?:er|re)s?)(?!\w)", _FLAGS), _conc_range),
    (re.compile(rf"{_START}(?P<n>{_NUM})\s*(?:mg\s?/\s?l|mg\s+per\s+lit(?:er|re)s?|milligrams?\s+per\s+lit(?:er|re)s?)(?!\w)", _FLAGS), _conc_single),
    (re.compile(rf"{_START}(?P<n>{_NUMW})(?:\s*-\s*|\s+)(?:(?:consecutive|business|working|calendar)\s+)?"
                rf"(?P<u>minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b", _FLAGS), _duration),
]

_GAZETTEERS: List[Tuple[str, str, str]] = [
    ("WATER_QUALITY", r"chlorine\s+residuals?", "chlorine_residual"),
    ("WATER_QUALITY", r"chlorine", "chlorine"),
    ("WATER_QUALITY", r"turbidity", "turbidity"),
    ("WATER_QUALITY", r"bacteri(?:a|al)", "bacteria"),
    ("WATER_QUALITY", r"e\.?\s?coli", "e_coli"),
    ("WATER_QUALITY", r"fluoride", "fluoride"),
    ("WATER_QUALITY", r"nitrates?", "nitrate"),
    ("WATER_QUALITY", r"ph", "ph"),
    ("FACILITY", r"reservoirs?", "reservoir"),
    ("FACILITY", r"treatment\s+(?:facilit(?:y|ies)|plants?)", "treatment_plant"),
    ("FACILITY", r"pump(?:ing)?\s+stations?", "pump_station"),
    ("FACILITY", r"hydrants?", "hydrant"),
    ("FACILITY", r"meters?", "meter"),
    ("FACILITY", r"trunk\s+mains?", "trunk_main"),
    ("FACILITY", r"valves?", "valve"),
]
_NOT_AFTER_NUMBER = r"(?<!\d)(?<!\d\s)(?<!\d-)"       # "10 meters" / "2-meter" are lengths, not water meters
for _kind, _pat, _canon in _GAZETTEERS:
    guard = _NOT_AFTER_NUMBER if _canon == "meter" else ""
    _PATTERNS.append((re.compile(rf"{guard}\b{_pat}\b", _FLAGS), _gazetteer(_kind, _canon)))


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Return the non-overlapping entities in `text`, in reading order."""
    if not text:
        return []
    found = []
    for pattern, build in _PATTERNS:
        for m in pattern.finditer(text):
            entity = build(m)
            entity["text"] = m.group(0)
            entity["start"], entity["end"] = m.start(), m.end()
            found.append(entity)
    # Longest match wins: earliest start first, and among equal starts the longer span.
    found.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    kept, last_end = [], -1
    for e in found:
        if e["start"] >= last_end:
            kept.append(e)
            last_end = e["end"]
    return kept


def entity_key(entity: Dict[str, Any]) -> Tuple[str, str]:
    """(type, normalized value): what two entities must share to count as the same fact."""
    return entity["type"], entity["normalized"]


def matching_entities(query_entities: List[Dict[str, Any]], passage_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Entities from the query that also appear in the passage (same type and normalised value)."""
    in_passage = {entity_key(e) for e in passage_entities}
    seen, out = set(), []
    for e in query_entities:
        key = entity_key(e)
        if key in in_passage and key not in seen:
            seen.add(key)
            out.append({"type": e["type"], "text": e["text"], "normalized": e["normalized"]})
    return out
