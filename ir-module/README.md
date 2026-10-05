# IR Module: Water Policy Search

Information Retrieval service for the Smart Water Management multi-agent system (IT3041 Information Retrieval and Web Analytics).

Given a question in ordinary words ("brown water coming from my tap", "my bill is three times higher than normal"), it finds the **passage** of the utility's policy documents that answers it, explains **why** it matched, and says so plainly when **nothing relevant** exists. It also pulls the **structured facts** out of the question and out of the answer (zones, drought stages, percentages, volumes, durations) with a rule-based named-entity recogniser.

## Where it fits

| Who calls it | Why |
|---|---|
| Router (`GET /ir/search`) | Attaches supporting documents to anomalies and to citizen complaints, and answers `policy_query` messages |
| Collector Agent (via the Router) | Fetches the SOP text to cite as evidence in an alert |

Port **8003**. The response keeps the fields the Router and Collector already read (`results[].document`, `.score`, `.snippet`). Everything new is additive, so no other service needed changing. The Router returns IR's JSON unchanged, so the new `entities` fields also appear in the Router's `policy_query`, `citizen_complaint` and anomaly responses. Nothing consumes them yet.

## How retrieval works

Each step is one classic IR technique:

| Step | What it does |
|---|---|
| **Text processing** | Lower-casing, tokenising (numbers are kept, so "stage 3" and "20 percent" are searchable), stop-word removal, **Porter stemming** ("leaking", "leaks", "leak" become one term) |
| **Passage index** | Each document is split into overlapping 2-sentence passages. The best passage per document is returned, so the snippet is the part that answers the question, not the top of the file. |
| **Ranking** | **Okapi BM25** (default): term-frequency saturation, length normalisation, IDF over documents, and a title boost (a simplified BM25F). **Cosine TF-IDF** is available for comparison (`method=tfidf`). |
| **Query expansion** | A small hand-built domain thesaurus maps customer words to policy words ("burst" to "leak", "brown" to "discoloration", "no water" to "outage, supply, tanker"). It lives in `DOMAIN_SYNONYMS` in `ir_module.py`. |
| **Query clean-up** | Drops filler and non-topic words: vague degree words ("higher", "normal"), zone names and severity labels the Router adds ("West", "high"), and "zone 4"-style references. |
| **Entity extraction (NER)** | `ner.py` finds zones, drought stages, percentages, volumes, pressures, durations, concentrations, water-quality parameters and facilities in the question and in every returned passage, and normalises them. It only **adds** output and never changes ranking. See [Named-entity recognition](#named-entity-recognition). |
| **Abstention** | Weak matches are dropped: a result needs enough absolute evidence *and* at least 30% of the best score. If nothing qualifies, the response says `abstained: true` instead of guessing. |
| **Explanation** | Each result lists the query words that matched, which matched only through the thesaurus, its raw score, and the words to highlight. |

`score` is normalised to 0 to 1 (the fraction of the maximum attainable BM25 score for that query), so consumers can threshold it whichever method is used.

## Quick start

Requires Python 3.12 or newer.

```powershell
cd ir-module
pip install -r requirements.txt
copy .env.example .env          # then set INTERNAL_API_KEY (same value as the other services)
python -m uvicorn main:app --port 8003
```

Expected startup: `Indexed 11 documents as 39 passages` and `Uvicorn running on http://127.0.0.1:8003`.

The service **refuses to start** without a valid API key (missing, under 16 characters, or still the `change-me...` placeholder). If `nltk` cannot be installed it falls back to a cruder stemmer and logs a warning, so it still runs, only less accurately.

Try it: open `index.html` in a browser, paste the key and search.

```powershell
$key = "<your key>"
Invoke-RestMethod -Uri http://127.0.0.1:8003/ir/search -Headers @{"X-API-Key"=$key} -Body @{query="brown water coming from my tap"}
```

## Configuration

Set in `ir-module/.env`. Only the key is required.

| Variable | Default | Meaning |
|---|---|---|
| `INTERNAL_API_KEY` | (required) | Shared key. `IR_API_KEY` overrides it for this service only. |
| `IR_BM25_K1` / `IR_BM25_B` | `1.5` / `0.75` | Standard BM25 parameters |
| `IR_PASSAGE_SENTENCES` | `2` | Sentences per passage |
| `IR_EXPANSION_WEIGHT` | `1.0` | Weight of thesaurus terms relative to the user's own words |
| `IR_TITLE_WEIGHT` | `2` | Extra weight for terms in a document title |
| `IR_IDF_LEVEL` | `document` | `document` or `passage` |
| `IR_MIN_BM25` | `1.0` | Evidence needed before a match is returned |
| `IR_REL_CUTOFF` | `0.3` | Drop results below this share of the best score |

## API

Every endpoint except `/ir/status` needs the header `X-API-Key: <key>`.

| Method | Path | Description |
|---|---|---|
| GET | `/ir/status` | Health check (open, used by the Router). Document, passage and vocabulary counts. |
| GET | `/ir/search` | `query` (required, at most 500 characters), `top_k` (1 to 10, default 3), `method` (`bm25` or `tfidf`), `expand` (`true` or `false`) |
| GET | `/ir/entities` | `text` (required, at most 500 characters). Named entities in any text, for agents that need the facts in a complaint or an alert message. Never logged. |
| GET | `/ir/documents/{name}` | Full text of one indexed document (for citing or grounding an answer) |
| POST | `/ir/reindex` | Re-read `data/documents/` after adding or editing files |

Example (`query = "my water bill is three times higher than normal"`, shortened):

```json
{
  "status": "ok",
  "query": "my water bill is three times higher than normal",
  "method": "bm25",
  "query_terms": ["water", "bill"],
  "expanded_terms": [],
  "unmatched_terms": [],
  "num_results": 2,
  "abstained": false,
  "results": [
    {
      "document": "meter_reading_and_billing_disputes.txt",
      "title": "Meter Reading and Billing Dispute Policy",
      "score": 0.5996,
      "raw_score": 2.3638,
      "method": "bm25",
      "passage_id": "meter_reading_and_billing_disputes.txt#1",
      "snippet": "When a meter cannot be read the bill is estimated and clearly marked as an estimate, ...",
      "matched_terms": ["bill"],
      "expanded_matches": [],
      "highlight_terms": ["bill"]
    }
  ]
}
```

When nothing is relevant (`query = "what is the capital of france"`): `"num_results": 0, "abstained": true, "note": "No sufficiently relevant policy passage was found for this query."`

## Named-entity recognition

`ner.py` is a rule-based recogniser (hand-written patterns and small gazetteers, longest match wins). It runs on the question, on every returned passage, and on request through `GET /ir/entities`. It needs no model or training data, so every extraction can be read and explained.

| Type | Examples | Normalised to |
|---|---|---|
| `ZONE` | "Zone 4", "Zone Four", "the west region" | `4` / `West` |
| `DROUGHT_STAGE` | "Stage 3", "Stage III" | `3` |
| `PERCENTAGE` | "20 percent", "42%", "twenty percent" | `20` |
| `VOLUME` | "5,000 liters per day", "11,398 L" | `5000 liters` (per `day`) |
| `PRESSURE` | "15 psi", "2 bar" | `15 psi` |
| `DURATION` | "24 hours", "7-day", "six month", "5 business days" | `24 hour` (plus `hours`) |
| `CONCENTRATION` | "between 0.2 and 4.0 milligrams per liter" | `0.2-4 mg/l` |
| `WATER_QUALITY` | "chlorine residual", "turbidity", "bacterial" | `chlorine_residual`, ... |
| `FACILITY` | "reservoir", "pumping station", "hydrant" | `reservoir`, `pump_station`, ... |

Example (`GET /ir/entities?text=Stage 3 restrictions in zone 2 within 48 hours`, shortened):

```json
{"num_entities": 3, "entities": [
  {"type": "DROUGHT_STAGE", "text": "Stage 3", "value": 3, "normalized": "3", "start": 0, "end": 7},
  {"type": "ZONE", "text": "zone 2", "value": "2", "normalized": "2", "start": 24, "end": 30},
  {"type": "DURATION", "text": "48 hours", "value": 48, "unit": "hour", "hours": 48.0, "normalized": "48 hour", "start": 38, "end": 46}
]}
```

In `/ir/search` responses, `entities` (top level) lists the facts in the **question**. Each result has `entities` (facts stated in that passage) and `entity_matches` (the question's facts that the passage also states, for example both mention `Stage 3`). The dashboard shows them as badges, green when they match.

**Evaluation** (`python eval/ner_eval.py`, details in `eval/ner_results.md`). Entity-level F1, matching on type and normalised value:

| Set | Texts | F1 | Trust |
|---|---|---|---|
| Development (11 documents + 18 questions) | 29 | 1.00 | Low: written with the extractor |
| Challenge (paraphrases and traps), first run **before** any fix | 30 | **0.91** | Honest estimate at that point |
| Challenge, after fixing its six errors | 30 | 1.00 | Low: fixed after seeing the errors |
| **Fresh** (new phrasings and traps, labelled first, run once after the fixes) | 22 | **0.98** | **Highest** |

The perfect development score says only that the code does what its author intended. The fresh-set 0.98 is the number to quote, and even that comes from a small set annotated by the person who designed the schema.

**Independent check** (`python eval/ner_independent_eval.py`, details in `eval/ner_independent_results.md`). The sets above were written to the extractor's own scope, which is why they score so high. As a second opinion, labels were written **before `ner.py` was read or run**, and deliberately broader (they also label incidents such as "leak" or "outage", dates such as "yesterday", and more facility words). Same type plus overlapping span counts as correct:

| Set | Texts | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| Independent, development (used to inspect errors) | 31 | 0.97 | 0.81 | 0.89 | On the types `ner.py` implements |
| Independent, **held-out** (run once, never used to change a rule) | 20 | **0.93** | **0.83** | **0.88** | On the types `ner.py` implements |

Counting **every** labelled type, including the two it does not implement, it recovers **53%** of the labelled entities (39 of 73 and 25 of 47). So it is precise where it applies, but it does not cover what a citizen complaint is mostly about. The recurring misses are: `INCIDENT` words (leak, burst, outage, no water), `DATE_REF` words (yesterday, Monday, last night), and facility words outside its list (pipe, pipeline, tap, tanker, sensor, dam, water main). The recurring false positives are a lowercase compass word read as a zone ("from the **west**", "drove **west**") and the verb "to **meter** the flow" read as a water meter.

**Zone values are not the Router's.** For "Zone 4" this module returns `4`, while the Router's own `extract_zone` maps numbers to names (`4` becomes `West`). Anything that combines the two must apply the same mapping.

**Limits:** it finds only the entity kinds above (not clock times, frequencies such as "twice weekly", or dates). **Zone names are ambiguous outside the domain**: "the South Pole" is read as the South zone. It is precise on the patterns it knows and blind to the rest, which is the trade-off of a rule-based approach.

## The document corpus

`data/documents/` holds **11 plain-text policy documents**. The first non-empty line of each file is its title. The four original documents (leak detection, drought response, water quality, complaint handling) came with the project. **The other seven were written for this assignment as illustrative sample content and are not real utility policies.** Thresholds and response times in them are placeholders. See `data/documents/README.md`.

To add a document: drop a `.txt` file into `data/documents/` and either restart the service or call `POST /ir/reindex`.

## Evaluation

Reproduce with `python evaluate.py`. It compares the original module with each improvement, using 34 development queries and 14 held-out queries. Relevance is judged at document level, top 3 results.

**Held-out queries** (written and labelled *before* tuning, never used to choose a setting; 12 in scope, 2 out of scope):

| System | Recall@3 | MRR | nDCG@3 | MAP@3 |
|---|---|---|---|---|
| Original module (whole-document TF-IDF) | 0.83 | 0.82 | 0.78 | 0.74 |
| BM25 or TF-IDF, whole document or passages (no expansion) | 0.79 | 0.88 | 0.80 | 0.77 |
| **Shipped: BM25, passages, stemming, expansion** | **0.88** | **0.96** | **0.88** | **0.85** |

**Development queries** (used while building; optimistic; 30 in scope, 4 out of scope): Recall@3 0.85 to 1.00, MRR 0.81 to 0.98, nDCG@3 0.81 to 0.98 (original to shipped). The full table, including the intermediate systems, is in `evaluation_results.md`.

How to read this honestly:

- The gain comes from the **stop-word fix, the passage index and query expansion**. **BM25 and TF-IDF tie** on this data. BM25 is kept as the default because it is the standard, better-founded ranking function, not because it measured better here.
- Only **one setting** mattered in a sensitivity study (`python eval/tune.py`): the expansion weight. Everything else moved the score by 0.01 or less, which is noise at this size.
- The corpus (11 documents) and query sets (48 queries) are small, and the same team wrote the documents, queries and thesaurus. Treat the numbers as an illustration, not a benchmark. Replace the sample documents and add real user queries before quoting them externally.
- Both sets contain out-of-scope queries. The module returned nothing for all of them, but there are only 6.

## What changed from the original module

| Area | Original | Now |
|---|---|---|
| Index | 4 documents, whole-document TF-IDF | 11 documents, passage-level BM25 (and TF-IDF) |
| Snippet | First 250 characters of the file, whatever the query | The best-matching passage |
| Words | Plain stop words, no stemming | Domain-safe stop list, Porter stemming, number tokens, thesaurus |
| Weak matches | Always returned | Abstains |
| Output | Document, score, snippet | Also: matched terms, thesaurus matches, passage id, raw score, highlights |
| Key | Hardcoded, non-constant-time comparison | From `.env`, constant-time, validated startup |
| Logging | (uvicorn printed full query text) | Query text never logged or stored |
| Entities | None | Rule-based NER on the question and each passage |
| Tests and evaluation | None | 127 tests, `evaluate.py` (retrieval) and `eval/ner_eval.py` (NER) |

Bugs found and fixed along the way: scikit-learn's stop-word list contains **"bill"** (so "water bill" silently lost its key word, in the original module too); "Zone 4" matched "4 hours" in unrelated documents; and uvicorn's access log printed every search query in full.

## Security and privacy

- API key from `.env`, constant-time comparison, validated at startup (see Quick start).
- Queries are cleaned (control characters removed, whitespace collapsed) and limited to 500 characters. `top_k` and `method` are validated.
- `/ir/documents/{name}` looks the name up in the index and never opens it as a file path, so `../` tricks return 404.
- **Privacy:** queries can contain personal details, because the Router forwards citizen complaints here. **Query text is never logged or stored.** The service logs only a short hash, the length and the result count, and query strings are redacted from uvicorn's access log (`GET /ir/search?[redacted]`).
- Dashboard renders document text with `textContent` (no HTML injection), and the key is typed in, never stored in the file.

Not implemented: HTTPS/TLS, rate limiting, restricted CORS (currently `*`).

## Testing

```powershell
pip install -r requirements-dev.txt
python -m pytest tests -q
```

127 tests: text processing, ranking maths, abstention, index reloading, the API (authentication, validation, path traversal), startup guards, the privacy guarantees, regression tests for each bug listed above, and the NER (each entity type and its normalisation, longest-match and overlap rules, traps that must not be extracted, hostile input that must not be slow, and a check that switching NER off leaves every ranking and score identical). No network needed.

## Known limitations

- **Keyword matching only.** It cannot understand meaning. "pipe burst near the school and water is flooding the street" still ranks *emergency supply* above the *leak procedure*, because "school" and "street" appear in other documents.
- **Held-out misses:** "is it a leak or just planned flushing", "how do residents find out the water is not safe to drink" and "do factories have to use less water in a crisis" each miss one relevant document.
- **Near-domain off-topic queries can slip through.** "how much does bottled water cost" returns the billing policy, and a bare "policy" matches a document with "Policy" in its title.
- **NER is rule-based**: it misses entity kinds not in its schema and reads a bare zone name such as "the South Pole" as a zone.
- **The thesaurus is hand-built** and small. It bridges only the vocabulary gaps someone thought of.
- The sample documents are illustrative, and evaluation uses document-level relevance only.

## Project structure

```
ir-module/
  main.py               FastAPI app: auth, validation, privacy-safe logging, endpoints
  ir_module.py          the retrieval engine (text processing, passage index, BM25/TF-IDF, expansion)
  ner.py                rule-based named-entity recognition
  evaluate.py           runs the evaluation (dev + held-out) and writes evaluation_results.md/.json
  evaluation_results.*  latest results
  index.html            search dashboard
  data/documents/       the policy corpus (.txt)
  eval/                 retrieval: queries.json, heldout_queries.json, tune.py, baseline_original.py
                        NER: ner_eval.py + its gold sets, ner_independent_eval.py + ner_independent_*.json
                        NER: ner_gold.json, ner_challenge.json, ner_fresh.json, ner_eval.py, ner_results.md
  tests/                pytest suite
  .env.example          configuration template
```

## Contributors

| Name | Student ID | Contribution |
|---|---|---|
| _TODO_ | _TODO_ | Collector Agent, IR Module |

## Acknowledgements

_TODO: list libraries (FastAPI, scikit-learn, NLTK), and any tools or AI assistance used, as your course policy requires._
