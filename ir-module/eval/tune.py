"""Sensitivity study: how much does each BM25 / index setting matter? Run: python eval/tune.py"""
import itertools, json, sys, warnings
warnings.filterwarnings("ignore")
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import evaluate as ev
from ir_module import IRConfig, WaterPolicyIR

Q = ev.load_queries()                       # DEV queries only
scope = [q for q in Q if q["kind"] != "out_of_scope"]; oos = [q for q in Q if q["kind"] == "out_of_scope"]
rows = []
for ew, tw, idf, k1, b, win in itertools.product([0.5, 1.0], [0, 2, 4], ["document", "passage"], [1.2, 1.5, 2.0], [0.5, 0.75], [1, 2, 3]):
    ir = WaterPolicyIR(ev.DOCS, IRConfig(expansion_weight=ew, title_weight=tw, idf_level=idf, k1=k1, b=b, window=win))
    fn = lambda q: [r["document"] for r in ir.search(q, 3, "bm25", True)["results"]]
    m = [ev.metrics_for(fn(q["query"]), set(q["relevant"])) for q in scope]
    mean = [sum(c) / len(m) for c in zip(*m)]
    fa = sum(1 for q in oos if fn(q["query"])) / len(oos)
    rows.append((mean[3], mean[2], mean[1], fa, dict(ew=ew, tw=tw, idf=idf, k1=k1, b=b, win=win)))
rows.sort(key=lambda r: (-r[0], -r[1]))
print("Hyperparameter sensitivity on the DEV queries (eval/queries.json). The held-out set is not used here.")
print("configs tried:", len(rows))
print("TOP 8 by nDCG@3 (dev set):")
for r in rows[:8]: print(f"  nDCG={r[0]:.3f} MRR={r[1]:.3f} R@3={r[2]:.3f} false_alarm={r[3]:.0%}  {r[4]}")
default = next(r for r in rows if r[4] == dict(ew=1.0, tw=2, idf="document", k1=1.5, b=0.75, win=2))
print(f"\nShipped defaults: nDCG={default[0]:.3f} MRR={default[1]:.3f} rank {rows.index(default)+1}/{len(rows)}")
print("Spread: best-vs-worst nDCG", round(rows[0][0],3), "vs", round(rows[-1][0],3))
# sensitivity per parameter (mean nDCG across other settings)
import collections
for key in ("ew", "tw", "idf", "k1", "b", "win"):
    agg = collections.defaultdict(list)
    for r in rows: agg[r[4][key]].append(r[0])
    print(f"  mean nDCG by {key:3s}:", {k: round(sum(v)/len(v), 3) for k, v in agg.items()})
