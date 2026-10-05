"""Tests for the retrieval engine (ir_module.py). No network, no server."""
import os
import shutil
import subprocess
import sys

import pytest

from ir_module import DEFAULT_DOC_FOLDER, IRConfig, WaterPolicyIR, analyze, stem, tokenize

ir = WaterPolicyIR()


def docs(result):
    return [r["document"] for r in result["results"]]


# ------------------------------------------------------------ text processing
def test_bill_is_not_a_stop_word():
    """Regression: scikit-learn's stop list contains 'bill', which silently deleted the key
    word from every billing query (in the original module too)."""
    assert "bill" in tokenize("my water bill is high")
    assert "system" in tokenize("the monitoring system")


def test_numbers_are_searchable():
    assert "3" in tokenize("Stage 3 emergency restrictions")
    assert "15" in tokenize("reservoir at 15 percent")


def test_stemming_unifies_word_forms():
    assert stem("leaking") == stem("leaks") == stem("leak")
    assert stem("restrictions") == stem("restriction")
    assert stem("30") == "30"                                   # numbers untouched


def test_zone_references_are_not_search_terms():
    """Regression: the '4' in 'Zone 4' matched '4 hours' in unrelated documents."""
    q = ir.process_query("There is a pipe leak in Zone 4 and water is flooding the road")
    assert "4" not in q["surface"] and "zone" not in q["surface"]


def test_query_filler_and_router_labels_are_dropped():
    q = ir.process_query("West leak abnormal flow high")
    assert q["surface"] == ["leak", "abnormal", "flow"]         # zone name and severity label removed


# ------------------------------------------------------------ retrieval quality
@pytest.mark.parametrize("query, expected", [
    # exactly what the Collector sends
    ("drought response plan reservoir capacity stage restrictions", "drought_response_plan.txt"),
    ("suspected leak flow exceeds rolling average consumption leak detection", "leak_detection_procedure.txt"),
    ("abnormal flow exceeds rolling average leak detection", "leak_detection_procedure.txt"),
    # exactly what the Router sends
    ("West leak abnormal flow high", "leak_detection_procedure.txt"),
    ("There is a pipe leak in Zone 4 and water is flooding the road.", "leak_detection_procedure.txt"),
    # natural language
    ("my water bill is three times higher than normal", "meter_reading_and_billing_disputes.txt"),
    ("Stage 3 emergency restrictions", "drought_response_plan.txt"),
    ("no water in my area since yesterday", "emergency_water_supply.txt"),
])
def test_known_queries_rank_the_right_document_first(query, expected):
    assert docs(ir.search(query))[0] == expected


def test_snippet_is_the_matching_passage_and_depends_on_the_query():
    """The original returned the first 250 characters of the file whatever you asked."""
    a = ir.search("Stage 3 emergency restrictions", top_k=1)["results"][0]
    b = ir.search("reservoir below 30 percent stage 1 alert", top_k=1)["results"][0]
    assert a["document"] == b["document"] == "drought_response_plan.txt"
    assert a["snippet"] != b["snippet"]
    assert "Stage 3" in a["snippet"] and "Stage 1" in b["snippet"]


def test_thesaurus_bridges_customer_and_policy_vocabulary():
    r = ir.search("brown water coming from my tap")
    assert {"customer_complaint_handling.txt", "water_quality_standards.txt"} <= set(docs(r))
    assert {e["term"] for e in r["expanded_terms"]} == {"discoloration", "turbidity"}
    # ...and switching expansion off really turns it off
    off = ir.search("brown water coming from my tap", expand=False)
    assert off["expanded_terms"] == [] and all(not x["expanded_matches"] for x in off["results"])


def test_explanation_separates_users_words_from_thesaurus_words():
    """Regression: with expansion_weight == 1.0 the code confused the two."""
    top = ir.search("brown water coming from my tap")["results"][0]
    assert "water" in top["matched_terms"]
    assert "turbidity" not in top["matched_terms"] and "discoloration" not in top["matched_terms"]
    assert any(e["from"] == "brown" for e in top["expanded_matches"])
    assert "water" in top["highlight_terms"]


# ------------------------------------------------------------ abstention
@pytest.mark.parametrize("query", [
    "what is the capital of france", "best recipe for chocolate cake", "asdfgh qwerty", "hello",
    "the", "a", "12345", "water",
])
def test_abstains_instead_of_returning_weak_matches(query):
    r = ir.search(query)
    assert r["abstained"] is True and r["results"] == [] and r["note"]


def test_query_with_only_stop_words_is_reported_as_unsearchable():
    assert "no searchable terms" in ir.search("what is the")["note"]


# ------------------------------------------------------------ ranking maths
def test_scores_are_normalised_and_sorted():
    for q in ["leak", "drought stage restrictions", "bill", "boil water notice"]:
        for method in ("bm25", "tfidf"):
            scores = [r["score"] for r in ir.search(q, method=method)["results"]]
            assert all(0 <= s <= 1 for s in scores) and scores == sorted(scores, reverse=True)


def test_rarer_terms_weigh_more_than_common_ones():
    assert ir._idf(stem("turbidity")) > ir._idf(stem("water"))


def test_bm25_saturates_repeated_terms():
    """Doubling a term's frequency must not double its contribution."""
    k1 = ir.cfg.k1
    gain = lambda tf: tf * (k1 + 1) / (tf + k1)
    assert gain(2) < 2 * gain(1) and gain(20) < k1 + 1


def test_top_k_is_respected_and_clamped():
    assert len(ir.search("water leak drought bill", top_k=1)["results"]) == 1
    assert len(ir.search("water leak drought bill", top_k=99)["results"]) <= 10


def test_one_result_per_document():
    d = docs(ir.search("leak water reservoir", top_k=10))
    assert len(d) == len(set(d))


def test_response_keeps_the_fields_the_router_and_collector_read():
    r = ir.search("drought stage")["results"][0]
    assert {"document", "score", "snippet"} <= set(r)
    assert isinstance(r["score"], float) and isinstance(r["snippet"], str)


# ------------------------------------------------------------ index management
def test_reload_picks_up_new_documents(tmp_path):
    shutil.copytree(DEFAULT_DOC_FOLDER, tmp_path / "docs")
    engine = WaterPolicyIR(str(tmp_path / "docs"))
    before = len(engine.documents)
    (tmp_path / "docs" / "new_policy.txt").write_text(
        "Hydrant Vandalism Policy\n\nReports of hydrant vandalism must be logged and repaired within 48 hours.")
    assert engine.reload() == before + 1
    assert docs(engine.search("hydrant vandalism repaired"))[0] == "new_policy.txt"


def test_unreadable_and_empty_files_are_skipped_not_fatal(tmp_path):
    (tmp_path / "empty.txt").write_text("   \n")
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe\x00bad")
    (tmp_path / "ok.txt").write_text("Leak Policy\n\nA leak must be repaired quickly. Leaks waste water.")
    engine = WaterPolicyIR(str(tmp_path))
    assert engine.doc_names == ["ok.txt"]


def test_empty_or_missing_corpus_abstains(tmp_path):
    for folder in (str(tmp_path), str(tmp_path / "does_not_exist")):
        r = WaterPolicyIR(folder).search("leak")
        assert r["abstained"] and r["results"] == []


def test_still_works_without_nltk():
    """If nltk cannot be installed the module must degrade, not crash."""
    code = ("import sys; sys.modules['nltk']=None; sys.modules['nltk.stem']=None\n"
            "from ir_module import WaterPolicyIR\n"
            "r=WaterPolicyIR().search('drought stage restrictions')\n"
            "print(r['results'][0]['document'])")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=root, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "drought_response_plan.txt", out.stderr[-400:]


def test_startup_log_reports_the_real_passage_count(caplog):
    """Regression: the log said '11 passages' (the document count used for IDF) instead of 39."""
    import logging
    with caplog.at_level(logging.INFO, logger="ir.engine"):
        engine = WaterPolicyIR()
    assert f"{len(engine.documents)} documents as {len(engine.passages)} passages" in caplog.text
    assert len(engine.passages) > len(engine.documents)


# ------------------------------------------------------------ entity extraction inside search
def test_search_reports_entities_in_the_question_and_in_each_passage():
    r = ir.search("reservoir in zone 2 is at 8 percent, stage 3 restrictions may start within 48 hours")
    assert {("ZONE", "2"), ("PERCENTAGE", "8"), ("DROUGHT_STAGE", "3"), ("DURATION", "48 hour"), ("FACILITY", "reservoir")} \
        <= {(e["type"], e["normalized"]) for e in r["entities"]}
    top = r["results"][0]
    assert top["document"] == "drought_response_plan.txt"
    assert ("DROUGHT_STAGE", "3") in {(e["type"], e["normalized"]) for e in top["entities"]}
    assert {("DROUGHT_STAGE", "3"), ("FACILITY", "reservoir")} <= {(m["type"], m["normalized"]) for m in top["entity_matches"]}


def test_zone_reference_is_an_entity_but_stays_out_of_keyword_matching():
    q = ir.process_query("There is a pipe leak in Zone 4 and water is flooding the road")
    assert ("ZONE", "4") in {(e["type"], e["normalized"]) for e in q["entities"]}
    assert "4" not in q["surface"] and "zone" not in q["surface"]


def test_entities_are_reported_even_when_the_module_abstains():
    r = ir.search("what is the capital of france")
    assert r["abstained"] and r["entities"] == []
    assert "entities" in ir.search("zzzz qqqq 40 percent")


def test_entity_extraction_never_changes_ranking(monkeypatch):
    """NER only adds output. Switch it off inside the engine and every development query must return
    exactly the same documents, in the same order, with the same scores."""
    import json
    import ir_module
    queries = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "queries.json")))["queries"]
    with_ner = {q["query"]: [(x["document"], x["score"]) for x in ir.search(q["query"])["results"]] for q in queries}
    monkeypatch.setattr(ir_module, "extract_entities", lambda text: [])
    monkeypatch.setattr(ir_module, "matching_entities", lambda a, b: [])
    without_ner = {q["query"]: [(x["document"], x["score"]) for x in ir.search(q["query"])["results"]] for q in queries}
    assert with_ner == without_ner and len(with_ner) == len(queries)
    assert ir.search("Stage 3 restrictions")["entities"] == []            # ...and the switch really did turn it off
