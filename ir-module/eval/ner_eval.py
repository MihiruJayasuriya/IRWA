"""
Evaluate the NER against hand-made gold annotations (eval/ner_gold.json).

    python eval/ner_eval.py

Matching is on (type, normalised value) sets per text, so "24 hours" and "24-hour" count as the
same fact and a fact mentioned twice counts once. Micro-averaged precision / recall / F1 overall and
per entity type; every miss and every wrong extraction is listed.

The gold set was written by the same person who designed the entity schema, before the extractor
existed, and it is small: treat the scores as an illustration, not a benchmark.
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from ner import entity_key, extract_entities  # noqa: E402


def load():
    with open(os.path.join(HERE, "eval", "ner_gold.json"), encoding="utf-8") as f:
        return json.load(f)


def texts(gold):
    for name, ents in gold["documents"].items():
        with open(os.path.join(HERE, "data", "documents", name), encoding="utf-8") as f:
            yield "document", name, f.read(), {tuple(e) for e in ents}
    for q in gold["queries"]:
        yield "query", q["text"], q["text"], {tuple(e) for e in q["gold"]}


def score(rows):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for _, _, _, gold, pred in rows:
        for t, _k in gold & pred: tp[t] += 1
        for t, _k in pred - gold: fp[t] += 1
        for t, _k in gold - pred: fn[t] += 1
    def prf(t=None):
        types = [t] if t else sorted(set(tp) | set(fp) | set(fn))
        a, b, c = (sum(d[x] for x in types) for d in (tp, fp, fn))
        p = a / (a + b) if a + b else 1.0
        r = a / (a + c) if a + c else 1.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0), a, b, c
    return prf, sorted(set(tp) | set(fp) | set(fn))


def report(rows, title):
    prf, types = score(rows)
    lines = [f"### {title}", "", "| Entity type | Precision | Recall | F1 | Correct | Wrong | Missed |", "|---|---|---|---|---|---|---|"]
    for t in types:
        p, r, f, a, b, c = prf(t)
        lines.append(f"| {t} | {p:.2f} | {r:.2f} | {f:.2f} | {a} | {b} | {c} |")
    p, r, f, a, b, c = prf()
    lines.append(f"| **All** | **{p:.2f}** | **{r:.2f}** | **{f:.2f}** | {a} | {b} | {c} |")
    return "\n".join(lines), (p, r, f)


def load_set(name):
    with open(os.path.join(HERE, "eval", name), encoding="utf-8") as f:
        return [(q["text"], q["text"], {tuple(e) for e in q["gold"]}) for q in json.load(f)["queries"]]


def run(items, kind):
    return [(kind, name, text, g, {entity_key(e) for e in extract_entities(text)}) for name, text, g in items]


def main():
    gold = load()
    dev = [(k, n, t, g, {entity_key(e) for e in extract_entities(t)}) for k, n, t, g in texts(gold)]
    challenge = run(load_set("ner_challenge.json"), "challenge")
    fresh = run(load_set("ner_fresh.json"), "fresh")

    out = ["# NER evaluation", "", "Reproduce with `python eval/ner_eval.py`. Entity-level precision / recall / F1 on (type, normalised value) sets per text.", ""]
    out += ["## Summary", "", "| Set | Texts | Precision | Recall | F1 | How much to trust it |", "|---|---|---|---|---|---|"]
    summary = [
        ("Development: 11 policy documents + 18 questions", dev, "Low: written with the extractor, so it is optimistic"),
        ("Challenge: paraphrases and traps, **after fixes**", challenge, "Low: the fixes were made after seeing these errors"),
        ("Fresh: new phrasings and traps, run once after the fixes", fresh, "**Highest**: labelled before it was run and never used to change the code"),
    ]
    for label, rows, trust in summary:
        _, (p, r, f) = report(rows, label)
        out.append(f"| {label} | {len(rows)} | {p:.2f} | {r:.2f} | {f:.2f} | {trust} |")
    out += ["", "**First run of the challenge set, before any fix** (recorded, not reproducible after the fixes): "
            "precision 0.94, recall 0.89, F1 0.91. It missed a word-number percentage (`twenty percent`), a word-number zone "
            "(`Zone Four`), a Roman-numeral stage (`Stage III`) and `mg per liter`, and wrongly read `10 meters` as a water meter "
            "and `Central Park` as a zone. All six were fixed. That 0.91 was the honest estimate before the fixes.", ""]
    for label, rows, _ in summary:
        table, _ = report(rows, label)
        out += [table, ""]
    errors = []
    for label, rows, _ in summary:
        for kind, name, _t, g, pred in rows:
            miss, wrong = sorted(g - pred), sorted(pred - g)
            if miss or wrong:
                errors.append(f"- [{label.split(':')[0]}] `{name[:70]}`: " + ("missed " + str(miss) + " " if miss else "") + ("wrong " + str(wrong) if wrong else ""))
    out += ["## Remaining errors", ""] + (errors or ["None."]) + ["",
            "## Limits", "",
            "- Rule-based: it finds only the entity kinds in the schema (zones, drought stages, percentages, volumes, pressures, durations, "
            "concentrations, water-quality parameters, facilities). Clock times, frequencies (`twice weekly`) and dates are out of scope.",
            "- **Zone names are ambiguous outside the domain**: `the South Pole` is read as the South zone.",
            "- The gold sets are small and were annotated by the person who designed the schema. Treat every score as an illustration.", ""]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(HERE, "eval", "ner_results.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
