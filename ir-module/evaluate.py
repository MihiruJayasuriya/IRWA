"""
evaluate.py - measure retrieval quality, and show which improvement helped.

    python evaluate.py

Systems compared (each adds one thing to the one before, except the last):
  A  Original module        whole-document TF-IDF, stop words, no stemming
  B  BM25, whole document   + stemming, number tokens, BM25 ranking
  C  TF-IDF, passages       passage-level index, TF-IDF cosine
  D  BM25, passages         passage-level index, BM25
  E  BM25, passages + expansion   = the module as shipped (default method)
  F  TF-IDF, passages + expansion the fair rival to E: same features, other ranking function

Metrics (top 3, document-level, binary relevance) over the 30 in-scope queries:
  P@3     share of returned documents that are relevant
  R@3     share of relevant documents that were returned
  MRR     1 / rank of the first relevant document
  nDCG@3  rank-aware gain (relevant documents near the top score higher)
  MAP@3   mean average precision
and over the 4 out-of-scope queries:
  False-alarm rate   share of out-of-scope queries that still returned something
                     (lower is better; the right answer is "nothing relevant")

Judgments live in eval/queries.json. They were written before any system was run,
but by the same people who wrote the documents and the thesaurus, on a small
corpus: treat the numbers as an illustration, not a benchmark.
"""

import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "eval"))

from ir_module import IRConfig, WaterPolicyIR                      # noqa: E402
from baseline_original import OriginalWaterPolicyIR                # noqa: E402

DOCS = os.path.join(HERE, "data", "documents")
K = 3


def load_queries(name="queries.json"):
    with open(os.path.join(HERE, "eval", name), encoding="utf-8") as f:
        return json.load(f)["queries"]


def build_systems():
    original = OriginalWaterPolicyIR(DOCS)
    doc_level = WaterPolicyIR(DOCS, IRConfig(window=999))          # one passage per document
    passages = WaterPolicyIR(DOCS)
    return {
        "A  Original (whole-doc TF-IDF)": lambda q: [r["document"] for r in original.search(q, top_k=K)],
        "B  BM25, whole document":        lambda q: [r["document"] for r in doc_level.search(q, K, "bm25", expand=False)["results"]],
        "C  TF-IDF, passages":            lambda q: [r["document"] for r in passages.search(q, K, "tfidf", expand=False)["results"]],
        "D  BM25, passages":              lambda q: [r["document"] for r in passages.search(q, K, "bm25", expand=False)["results"]],
        "E  BM25, passages + expansion (shipped)":
                                          lambda q: [r["document"] for r in passages.search(q, K, "bm25", expand=True)["results"]],
        "F  TF-IDF, passages + expansion": lambda q: [r["document"] for r in passages.search(q, K, "tfidf", expand=True)["results"]],
    }


def metrics_for(ranked, relevant):
    ranked = ranked[:K]
    rel_flags = [1 if d in relevant else 0 for d in ranked]
    hits = sum(rel_flags)
    p = hits / K
    r = hits / len(relevant)
    rr = next((1 / (i + 1) for i, f in enumerate(rel_flags) if f), 0.0)
    dcg = sum(f / math.log2(i + 2) for i, f in enumerate(rel_flags))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), K)))
    ap, seen = 0.0, 0
    for i, f in enumerate(rel_flags):
        if f:
            seen += 1
            ap += seen / (i + 1)
    ap /= min(len(relevant), K)
    return p, r, rr, dcg / idcg, ap


def evaluate(systems, queries):
    scope = [q for q in queries if q["kind"] != "out_of_scope"]
    oos = [q for q in queries if q["kind"] == "out_of_scope"]
    table = {}
    for name, fn in systems.items():
        rows = [metrics_for(fn(q["query"]), set(q["relevant"])) for q in scope]
        mean = [sum(col) / len(rows) for col in zip(*rows)]
        false_alarm = sum(1 for q in oos if fn(q["query"])) / len(oos)
        by_kind = {}
        for kind in ("system", "natural"):
            sub = [metrics_for(fn(q["query"]), set(q["relevant"])) for q in scope if q["kind"] == kind]
            by_kind[kind] = (sum(r[2] for r in sub) / len(sub)) if sub else None      # MRR per kind
        table[name] = {"P@3": mean[0], "R@3": mean[1], "MRR": mean[2], "nDCG@3": mean[3],
                       "MAP@3": mean[4], "false_alarm": false_alarm,
                       "MRR_system_queries": by_kind["system"], "MRR_natural_queries": by_kind["natural"]}
    return table, len(scope), len(oos)


def render(table, n_scope, n_oos, failures, title="Development queries", source="eval/queries.json"):
    cols = ["P@3", "R@3", "MRR", "nDCG@3", "MAP@3"]
    lines = [f"## {title}\n",
             f"{n_scope} in-scope queries and {n_oos} out-of-scope queries, 11 sample documents "
             f"(judgments in `{source}`).\n",
             "| System | " + " | ".join(cols) + " | False-alarm rate |", "|---|" + "---|" * (len(cols) + 1)]
    for name, m in table.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + f" | {m['false_alarm']:.0%} |")
    lines += ["", "MRR split by query type:", "", "| System | System-generated queries | Natural-language queries |", "|---|---|---|"]
    for name, m in table.items():
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        lines.append(f"| {name} | {fmt(m['MRR_system_queries'])} | {fmt(m['MRR_natural_queries'])} |")
    if failures:
        lines += ["", "## Queries the shipped system still gets wrong or only partly right", ""]
        lines += failures
    return "\n".join(lines)


CAVEATS = """## How to read these numbers

- **Development queries** (34) were used while building the module: to find bugs, and to choose the one setting
  that mattered (`expansion_weight`, see `eval/tune.py`). Their scores are optimistic.
- **Held-out queries** (14) were written and labelled *before* tuning and were never used to choose a setting.
  They were run after the engine was finished, and again after two final bug fixes (scores identical).
  They are the fairer estimate of how the module behaves on unseen questions.
- The corpus is small (11 sample documents) and the sets are small: differences of a few points are not significant.
- Queries, documents and the domain thesaurus were written by the same team, so expansion gains are probably
  optimistic. Replace the sample documents and add real user queries before quoting these numbers externally.
- Relevance is judged at document level, not passage level."""


def section(queries, shipped_name, title, source):
    systems = build_systems()
    table, n_scope, n_oos = evaluate(systems, queries)
    shipped = systems[shipped_name]
    failures = []
    for q in queries:
        got = shipped(q["query"])
        rel = set(q["relevant"])
        if q["kind"] == "out_of_scope":
            if got:
                failures.append(f"- `{q['query']}` (out of scope) returned {got}")
        elif not rel.issubset(set(got)) or (got and got[0] not in rel):
            failures.append(f"- `{q['query']}`: expected {sorted(rel)}, got {got}")
    return table, render(table, n_scope, n_oos, failures, title, source)


def main():
    shipped = "E  BM25, passages + expansion (shipped)"
    dev_table, dev_text = section(load_queries("queries.json"), shipped, "Development queries", "eval/queries.json")
    held_table, held_text = section(load_queries("heldout_queries.json"), shipped, "Held-out queries (labelled before tuning, never used to choose a setting)",
                                    "eval/heldout_queries.json")
    text = "# IR evaluation results\n\nReproduce with `python evaluate.py`.\n\n" + held_text + "\n\n" + dev_text + "\n\n" + CAVEATS
    print(text)
    with open(os.path.join(HERE, "evaluation_results.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(HERE, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump({"held_out": held_table, "development": dev_table}, f, indent=1)


if __name__ == "__main__":
    main()
