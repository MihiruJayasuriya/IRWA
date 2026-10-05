"""
ner_independent_eval.py - score ner.py against labels that were written independently of it.

    python eval/ner_independent_eval.py

The gold sets (ner_independent_dev.json, ner_independent_heldout.json) were labelled before
ner.py had been read or run, and are deliberately BROADER than what the extractor implements.
So the report has two views:
  * "implemented types": precision / recall / F1 on the types ner.py claims to handle
  * "all labelled types": how much of everything that was labelled it recovers, counting the
    types it does not implement (INCIDENT, DATE_REF) as misses.

Matching is lenient: same entity type and overlapping character span (so "West" vs "West zone"
counts). "strict recall" additionally requires identical text.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ner import extract_entities  # noqa: E402

IMPLEMENTED = {"ZONE", "DROUGHT_STAGE", "PERCENTAGE", "VOLUME", "DURATION", "FACILITY"}


def locate(text, gold_text, used):
    low, g, i = text.lower(), gold_text.lower(), 0
    while True:
        j = low.find(g, i)
        if j < 0:
            return None
        if (j, j + len(g)) not in used:
            used.add((j, j + len(g)))
            return (j, j + len(g))
        i = j + 1


def score(path):
    sentences = json.load(open(path, encoding="utf-8"))["sentences"]
    tp, fp, fn, strict = (collections.Counter() for _ in range(4))
    errors = []
    for s in sentences:
        text, used = s["text"], set()
        gold = [(g["type"], locate(text, g["text"], used), g["text"]) for g in s["entities"]]
        pred = [(e["type"], (e["start"], e["end"]), e["text"]) for e in extract_entities(text)]
        taken = set()
        for gtype, gspan, gtext in gold:
            hit = next((k for k, (ptype, pspan, _) in enumerate(pred)
                        if k not in taken and ptype == gtype and gspan and pspan[0] < gspan[1] and gspan[0] < pspan[1]), None)
            if hit is None:
                fn[gtype] += 1
                errors.append(("missed", s["id"], gtype, gtext, text))
            else:
                taken.add(hit)
                tp[gtype] += 1
                strict[gtype] += pred[hit][2].lower() == gtext.lower()
        for k, (ptype, _, ptext) in enumerate(pred):
            if k not in taken:
                fp[ptype] += 1
                errors.append(("extra", s["id"], ptype, ptext, text))
    return len(sentences), tp, fp, fn, strict, errors


def section(title, path):
    n, tp, fp, fn, strict, errors = score(path)
    lines = [f"## {title}", "", f"{n} texts. Same type + overlapping span counts as correct.", "",
             "| Type | Gold | Precision | Recall | F1 | Strict recall |", "|---|---|---|---|---|---|"]
    agg = collections.Counter()
    for t in sorted(set(tp) | set(fp) | set(fn), key=lambda t: (t not in IMPLEMENTED, t)):
        g = tp[t] + fn[t]
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else None
        r = tp[t] / g if g else None
        f = 2 * p * r / (p + r) if p and r else None
        cell = lambda v: "n/a" if v is None else f"{v:.2f}"
        note = "" if t in IMPLEMENTED else (" (not implemented)" if g else " (implemented, but not labelled in this set)")
        lines.append(f"| {t}{note} | {g} | {cell(p)} | {cell(r)} | {cell(f)} | {cell(strict[t] / g if g else None)} |")
        if t in IMPLEMENTED:
            agg.update(tp=tp[t], fp=fp[t], fn=fn[t])
    P, R = agg["tp"] / (agg["tp"] + agg["fp"]), agg["tp"] / (agg["tp"] + agg["fn"])
    total_gold = sum(tp.values()) + sum(fn.values())
    lines += ["", f"**Implemented types:** precision {P:.2f}, recall {R:.2f}, F1 {2 * P * R / (P + R):.2f} "
                  f"({agg['tp']} correct, {agg['fp']} extra, {agg['fn']} missed).",
              f"**All labelled types:** {sum(tp.values())} of {total_gold} labelled entities recovered ({sum(tp.values()) / total_gold:.0%}).", "",
              "Errors on implemented types:", ""]
    lines += [f"- {k} `{x}` ({t}) in: {text}" for k, _, t, x, text in errors if t in IMPLEMENTED] or ["- none"]
    return "\n".join(lines)


def main():
    out = ["# NER: independent evaluation", "",
           "Reproduce with `python eval/ner_independent_eval.py`. Labels were written before `ner.py` was read or run, and are broader "
           "than its scope, so this shows both its accuracy where it applies and where its coverage stops. Small sets: treat differences "
           "of a few points as noise. Types the extractor implements but these labels do not cover (WATER_QUALITY, PRESSURE, CONCENTRATION) are "
           "shown for information and left out of the precision figures. Strict recall is lower than lenient recall where span conventions differ "
           "(for example the extractor includes \"per day\" in a VOLUME span).", "",
           section("Development set", os.path.join(HERE, "ner_independent_dev.json")), "",
           section("Held-out set", os.path.join(HERE, "ner_independent_heldout.json"))]
    text = "\n".join(out) + "\n"
    print(text)
    open(os.path.join(HERE, "ner_independent_results.md"), "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
