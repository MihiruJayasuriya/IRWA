"""
ir_module.py - retrieval engine for the water-policy documents.

What the original module did: TF-IDF over WHOLE documents, cosine similarity,
"snippet" = first 250 characters of the document, no stemming, always returns
whatever scored above zero.

What this version does (each step is one classic IR technique):

  1. Text processing   lower-casing, tokenising (numbers kept, so "stage 3"
                       and "15 percent" are searchable), stop-word removal,
                       Porter stemming ("leaking" / "leaks" -> "leak").
  2. Passage indexing  every document is split into overlapping 2-sentence
                       passages, and the best passage is retrieved. The snippet
                       shown is therefore the passage that answers the query,
                       not the top of the file.
  3. Ranking           Okapi BM25 (default) with term-frequency saturation and
                       length normalisation, or cosine TF-IDF for comparison.
  4. Query expansion   a small hand-built domain thesaurus maps the words
                       customers use ("burst", "brown water") to the words the
                       policies use ("leak", "discoloration"). The mapping is
                       curated by hand, so a mapped term counts as much as the
                       user's own word (see IRConfig.expansion_weight).
  5. Abstention        weak matches are dropped. If nothing is relevant the
                       module says so instead of returning noise.
  6. Explainability    every result lists the query terms that matched, which
                       terms only matched through expansion, and its raw score.
  7. Entity extraction (ner.py) the question and every returned passage are scanned
                       for structured facts (zones, drought stages, percentages,
                       volumes, durations, ...). This only ADDS output: it never
                       changes which documents are returned or how they are ranked.

Every response keeps the fields the Router and Collector already read
(`results[].document`, `.score`, `.snippet`); the new fields are additions.
`score` is normalised to 0-1 so consumers can threshold it whatever the method.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ner import extract_entities, matching_entities

log = logging.getLogger("ir.engine")

try:                                    # Porter stemmer from NLTK (pure Python, no data download needed)
    from nltk.stem import PorterStemmer
    _porter = PorterStemmer()

    def _stem_word(token: str) -> str:
        return _porter.stem(token)
except Exception:                       # pragma: no cover - only if nltk cannot be installed
    log.warning("nltk is not available: using a crude suffix-stripping stemmer instead of Porter. "
                "Retrieval still works but is less accurate. Fix with: pip install nltk")

    def _stem_word(token: str) -> str:
        for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "edly", "ed", "ies", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOC_FOLDER = os.path.join(BASE_DIR, "data", "documents")

# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# "zone 4", "area west", "region central": entity references, not topics. The Router extracts them
# separately; left in the query, the "4" would match "4 hours" in unrelated policies.
_ZONE_REF_RE = re.compile(r"\b(?:zone|area|region)\s+(?:\d+|north|south|east|west|central)\b")


@lru_cache(maxsize=None)
def stem(token: str) -> str:
    """Porter stem; numbers are left alone."""
    return token if token[0].isdigit() else _stem_word(token)


# scikit-learn's English stop list contains some words that carry real meaning for a water
# utility - notably "bill" (so a search for "water bill" silently lost its key word), plus
# "system", "amount", "full", "empty", "fire", "call". They are removed from the stop list.
DOMAIN_WORDS_NOT_STOPPED = frozenset({
    "bill", "system", "amount", "full", "empty", "fire", "call", "found", "part", "front",
    "back", "top", "bottom", "detail", "fill", "move", "keep",
})
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - DOMAIN_WORDS_NOT_STOPPED


def tokenize(text: str) -> List[str]:
    """Lower-case content tokens: stop words and stray single letters removed."""
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def analyze(text: str) -> List[str]:
    return [stem(t) for t in tokenize(text)]


# Hand-built domain thesaurus: vocabulary people use -> vocabulary the policy
# documents use. Keys may be single words or short phrases. Deliberately small
# and explicit, so every expansion can be explained (and challenged) in the viva.
DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    "burst": ["leak"], "flooding": ["leak"], "gushing": ["leak"], "spraying": ["leak"],
    "abnormal": ["anomaly"], "unusual flow": ["anomaly"],
    "brown": ["discoloration", "turbidity"], "dirty": ["discoloration", "turbidity"],
    "muddy": ["discoloration", "turbidity"], "cloudy": ["discoloration", "turbidity"],
    "rusty": ["discoloration", "turbidity"],
    "smell": ["odor"], "smells": ["odor"], "odour": ["odor"], "stinks": ["odor"], "taste": ["odor"],
    "unsafe": ["contamination"], "polluted": ["contamination"], "sick": ["contamination"],
    "invoice": ["bill"], "overcharged": ["bill", "charge"],
    "lawn": ["garden"], "lawns": ["garden"], "yard": ["garden"],
    "no water": ["outage", "supply", "tanker"], "without water": ["outage", "supply", "tanker"],
    "water cut": ["outage", "supply"], "dry taps": ["outage", "supply"],
}


# Words that describe HOW someone asks rather than WHAT they ask about (degree, comparison,
# vague time). Dropped from queries only, never from documents. On a corpus this small IDF
# cannot tell them from topic words: "normal" or "higher" occur in one document each, so they
# look highly informative. Kept short and generic on purpose.
QUERY_FILLER = frozenset({
    "higher", "lower", "highest", "lowest", "normal", "normally", "usual", "usually", "typical",
    "times", "lot", "lots", "really", "quite", "bit", "yesterday", "today", "tomorrow", "tonight",
    "ago", "recently", "currently", "strange", "weird", "wrong", "okay", "ok", "thing", "things",
    # Not topics: the Router builds anomaly queries like "<zone> leak abnormal flow <severity>",
    # so zone names (entities) and the Analysis agent's severity labels are dropped too.
    "north", "south", "east", "west", "central",
    "high", "moderate", "critical", "severe",
})


@dataclass(frozen=True)
class IRConfig:
    k1: float = 1.5                 # BM25 term-frequency saturation
    b: float = 0.75                 # BM25 length normalisation
    window: int = 2                 # sentences per passage
    expansion_weight: float = 1.0   # weight of a thesaurus term relative to the user's own word (1.0 = equal)
    title_weight: float = 2.0       # a term in the title counts this many extra times (simplified BM25F)
    idf_level: str = "document"     # "document": rarity across documents; "passage": across passages
    min_raw_bm25: float = 1.0       # absolute evidence needed: below this a match is noise
    min_cosine: float = 0.08        # same idea for the TF-IDF method
    rel_cutoff: float = 0.3         # drop results scoring under 30% of the best result
    snippet_chars: int = 400

    @classmethod
    def from_env(cls) -> "IRConfig":
        def num(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, default))
            except (TypeError, ValueError):
                return default
        return cls(
            k1=num("IR_BM25_K1", cls.k1), b=num("IR_BM25_B", cls.b),
            window=int(num("IR_PASSAGE_SENTENCES", cls.window)),
            expansion_weight=num("IR_EXPANSION_WEIGHT", cls.expansion_weight),
            title_weight=num("IR_TITLE_WEIGHT", cls.title_weight),
            idf_level=os.getenv("IR_IDF_LEVEL", cls.idf_level),
            min_raw_bm25=num("IR_MIN_BM25", cls.min_raw_bm25),
            min_cosine=num("IR_MIN_COSINE", cls.min_cosine),
            rel_cutoff=num("IR_REL_CUTOFF", cls.rel_cutoff),
        )


@dataclass
class Passage:
    doc_index: int
    doc: str
    title: str
    number: int
    text: str                 # what is shown to the user
    terms: Counter            # stemmed terms of title + text (used for matching / explanation)
    body_terms: Counter       # stemmed terms of the passage text only
    title_terms: Counter      # stemmed terms of the document title
    length: int


def split_sentences(body: str) -> List[str]:
    body = re.sub(r"\s+", " ", body).strip()
    return [s for s in _SENTENCE_RE.split(body) if s]


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class WaterPolicyIR:
    def __init__(self, doc_folder: str = DEFAULT_DOC_FOLDER, config: Optional[IRConfig] = None):
        self.doc_folder = doc_folder
        self.cfg = config or IRConfig()
        self.reload()

    # -- indexing --------------------------------------------------------
    def reload(self) -> int:
        """(Re)read the .txt files and rebuild every index. Returns the document count."""
        self.doc_names: List[str] = []
        self.documents: List[str] = []
        self.titles: List[str] = []
        self.passages: List[Passage] = []
        self.N = 0

        if os.path.isdir(self.doc_folder):
            for fname in sorted(os.listdir(self.doc_folder)):
                if not fname.endswith(".txt"):
                    continue
                try:
                    with open(os.path.join(self.doc_folder, fname), "r", encoding="utf-8") as f:
                        raw = f.read()
                except (OSError, UnicodeDecodeError) as exc:
                    log.warning("Skipping unreadable document %s: %s", fname, exc)
                    continue
                if raw.strip():
                    self._add_document(fname, raw)

        # Vocabulary and document frequency. Document-level df is the default: indexing the
        # title with every passage would otherwise make a document's own topic word look
        # common and lower its weight.
        self.vocab = set()
        self.df: Counter = Counter()
        if self.cfg.idf_level == "passage":
            for p in self.passages:
                self.df.update(p.terms.keys())
            self.N = len(self.passages)
        else:
            per_doc: Dict[int, set] = {}
            for p in self.passages:
                per_doc.setdefault(p.doc_index, set()).update(p.terms.keys())
            for terms in per_doc.values():
                self.df.update(terms)
            self.N = len(per_doc)
        for p in self.passages:
            self.vocab.update(p.terms.keys())
        self.avgdl = (sum(p.length for p in self.passages) / len(self.passages)) if self.passages else 0.0
        self._tfidf = None            # built lazily, only if the tfidf method is used
        log.info("Indexed %d documents as %d passages (%d distinct terms)",
                 len(self.documents), len(self.passages), len(self.vocab))
        return len(self.documents)

    def _add_document(self, fname: str, raw: str) -> None:
        lines = [ln.strip() for ln in raw.strip().splitlines()]
        non_empty = [ln for ln in lines if ln]
        title = non_empty[0] if non_empty else fname
        body = " ".join(non_empty[1:]) or title
        doc_index = len(self.documents)
        self.doc_names.append(fname)
        self.documents.append(raw)
        self.titles.append(title)

        title_terms = Counter(analyze(title))
        sentences = split_sentences(body)
        w = max(1, self.cfg.window)
        for n, i in enumerate(range(0, max(1, len(sentences) - w + 1))):
            text = " ".join(sentences[i:i + w])
            body_terms = Counter(analyze(text))
            self.passages.append(Passage(
                doc_index, fname, title, n, text, body_terms + title_terms,
                body_terms, title_terms, sum(body_terms.values()) + sum(title_terms.values())))

    # -- query processing ------------------------------------------------
    def process_query(self, query: str, expand: bool = True) -> Dict[str, Any]:
        """Turn raw text into weighted stemmed terms, remembering where each came from."""
        entities = extract_entities(query)            # read from the original text: "Zone 4" is an entity here,
        lowered = re.sub(r"\s+", " ", _ZONE_REF_RE.sub(" ", query.lower())).strip()   # but is kept out of keyword matching
        surface = [t for t in tokenize(lowered) if t not in QUERY_FILLER]
        terms: Dict[str, float] = {}
        surface_of: Dict[str, str] = {}
        for tok in surface:
            s = stem(tok)
            terms[s] = 1.0
            surface_of.setdefault(s, tok)
        user_stems = set(terms)                       # the user's own words (vs thesaurus additions)
        expansion_stems: set = set()

        expansions: List[Dict[str, str]] = []
        if expand:
            for trigger, targets in DOMAIN_SYNONYMS.items():
                if re.search(rf"\b{re.escape(trigger)}\b", lowered) is None:
                    continue
                for target in targets:
                    ts = stem(target)
                    if ts in terms or ts not in self.vocab:
                        continue          # already asked for, or not in the corpus: useless
                    terms[ts] = self.cfg.expansion_weight
                    surface_of[ts] = target
                    expansion_stems.add(ts)
                    expansions.append({"term": target, "from": trigger})

        unmatched = [surface_of[s] for s in user_stems if s not in self.vocab]
        return {"terms": terms, "surface_of": surface_of, "surface": surface, "expansions": expansions,
                "unmatched": unmatched, "user_stems": user_stems, "expansion_stems": expansion_stems,
                "entities": entities}

    # -- ranking ---------------------------------------------------------
    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def _score_bm25(self, terms: Dict[str, float]) -> List[Tuple[Passage, float, float]]:
        k1, b = self.cfg.k1, self.cfg.b
        # Upper bound of BM25 for this query (term frequency -> infinity, empty passage):
        # dividing by it gives a 0-1 "fraction of the attainable score".
        bound = sum(w * self._idf(t) * (k1 + 1) for t, w in terms.items() if t in self.vocab)
        out = []
        for p in self.passages:
            norm = k1 * (1 - b + b * p.length / self.avgdl)
            raw = 0.0
            for t, w in terms.items():
                # simplified BM25F: title occurrences count `title_weight` extra times
                tf = p.body_terms.get(t, 0) + (1 + self.cfg.title_weight) * p.title_terms.get(t, 0)
                if tf:
                    raw += w * self._idf(t) * tf * (k1 + 1) / (tf + norm)
            if raw > 0:
                out.append((p, raw, raw / bound if bound else 0.0))
        return out

    def _ensure_tfidf(self) -> None:
        if self._tfidf is None:
            vec = TfidfVectorizer(analyzer=str.split, sublinear_tf=True)
            docs = [" ".join(p.terms.elements()) for p in self.passages]
            self._tfidf = (vec, vec.fit_transform(docs))

    def _score_tfidf(self, terms: Dict[str, float]) -> List[Tuple[Passage, float, float]]:
        self._ensure_tfidf()
        vec, matrix = self._tfidf
        qvec = vec.transform([" ".join(terms.keys())])
        sims = cosine_similarity(qvec, matrix).ravel()
        return [(p, float(s), float(s)) for p, s in zip(self.passages, sims) if s > 0]

    # -- public search ---------------------------------------------------
    def search(self, query: str, top_k: int = 3, method: str = "bm25", expand: bool = True) -> Dict[str, Any]:
        method = method if method in ("bm25", "tfidf") else "bm25"
        top_k = max(1, min(int(top_k), 10))
        q = self.process_query(query, expand=expand)
        base = {"method": method, "query_terms": q["surface"], "expanded_terms": q["expansions"],
                "unmatched_terms": q["unmatched"], "entities": q["entities"]}

        if not self.passages or not q["terms"]:
            return {**base, "num_results": 0, "results": [], "abstained": True,
                    "note": "The query contains no searchable terms." if self.passages else "No documents are indexed."}

        scored = self._score_bm25(q["terms"]) if method == "bm25" else self._score_tfidf(q["terms"])

        # Keep each document's best passage ("MaxP"), so the caller still gets one hit per document.
        best: Dict[int, Tuple[Passage, float, float]] = {}
        for p, raw, norm in scored:
            if p.doc_index not in best or raw > best[p.doc_index][1]:
                best[p.doc_index] = (p, raw, norm)
        ranked = sorted(best.values(), key=lambda x: x[1], reverse=True)

        # Abstention: absolute evidence first, then relative to the best hit.
        floor = self.cfg.min_raw_bm25 if method == "bm25" else self.cfg.min_cosine
        ranked = [r for r in ranked if r[1] >= floor]
        if ranked:
            top = ranked[0][1]
            ranked = [r for r in ranked if r[1] >= self.cfg.rel_cutoff * top]
        ranked = ranked[:top_k]

        results = [self._describe(p, raw, norm, q, method) for p, raw, norm in ranked]
        out = {**base, "num_results": len(results), "results": results, "abstained": not results}
        if not results:
            out["note"] = "No sufficiently relevant policy passage was found for this query."
        return out

    def _describe(self, p: Passage, raw: float, norm: float, q: Dict[str, Any], method: str) -> Dict[str, Any]:
        own = [q["surface_of"][s] for s in q["terms"] if s in q["user_stems"] and s in p.terms]
        via = {e["term"]: e["from"] for e in q["expansions"]}
        expanded = [{"term": q["surface_of"][s], "from": via.get(q["surface_of"][s], "")}
                    for s in q["terms"] if s in q["expansion_stems"] and s in p.terms]
        matched_stems = {stem(t) for t in own} | {stem(e["term"]) for e in expanded}
        highlight = sorted({t for t in _TOKEN_RE.findall(p.text.lower()) if stem(t) in matched_stems})
        text = p.text if len(p.text) <= self.cfg.snippet_chars else \
            p.text[: self.cfg.snippet_chars].rsplit(" ", 1)[0] + "..."
        passage_entities = extract_entities(text)         # offsets refer to the snippet as returned
        return {
            "document": p.doc,
            "title": p.title,
            "score": round(norm, 4),          # 0-1, comparable across methods
            "raw_score": round(raw, 4),       # the method's own score
            "method": method,
            "passage_id": f"{p.doc}#{p.number}",
            "snippet": text,                  # the best-matching passage, not the top of the file
            "matched_terms": own,
            "expanded_matches": expanded,
            "highlight_terms": highlight,
            "entities": passage_entities,                                        # facts stated in this passage
            "entity_matches": matching_entities(q["entities"], passage_entities),  # ...that the question also mentioned
        }
