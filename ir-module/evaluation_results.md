# IR evaluation results

Reproduce with `python evaluate.py`.

## Held-out queries (labelled before tuning, never used to choose a setting)

12 in-scope queries and 2 out-of-scope queries, 11 sample documents (judgments in `eval/heldout_queries.json`).

| System | P@3 | R@3 | MRR | nDCG@3 | MAP@3 | False-alarm rate |
|---|---|---|---|---|---|---|
| A  Original (whole-doc TF-IDF) | 0.36 | 0.83 | 0.82 | 0.78 | 0.74 | 0% |
| B  BM25, whole document | 0.33 | 0.79 | 0.88 | 0.80 | 0.77 | 0% |
| C  TF-IDF, passages | 0.33 | 0.79 | 0.88 | 0.80 | 0.77 | 0% |
| D  BM25, passages | 0.33 | 0.79 | 0.88 | 0.80 | 0.77 | 0% |
| E  BM25, passages + expansion (shipped) | 0.39 | 0.88 | 0.96 | 0.88 | 0.85 | 0% |
| F  TF-IDF, passages + expansion | 0.39 | 0.88 | 0.96 | 0.88 | 0.85 | 0% |

MRR split by query type:

| System | System-generated queries | Natural-language queries |
|---|---|---|
| A  Original (whole-doc TF-IDF) | n/a | 0.82 |
| B  BM25, whole document | n/a | 0.88 |
| C  TF-IDF, passages | n/a | 0.88 |
| D  BM25, passages | n/a | 0.88 |
| E  BM25, passages + expansion (shipped) | n/a | 0.96 |
| F  TF-IDF, passages + expansion | n/a | 0.96 |

## Queries the shipped system still gets wrong or only partly right

- `is it a leak or just planned flushing`: expected ['leak_detection_procedure.txt', 'pipe_maintenance_and_flushing.txt'], got ['pipe_maintenance_and_flushing.txt', 'drought_response_plan.txt']
- `how do residents find out the water is not safe to drink`: expected ['public_notification_boil_water.txt', 'water_quality_standards.txt'], got ['emergency_water_supply.txt', 'public_notification_boil_water.txt', 'drought_response_plan.txt']
- `do factories have to use less water in a crisis`: expected ['drought_response_plan.txt', 'water_conservation_measures.txt'], got ['drought_response_plan.txt', 'sensor_calibration_and_telemetry.txt']

## Development queries

30 in-scope queries and 4 out-of-scope queries, 11 sample documents (judgments in `eval/queries.json`).

| System | P@3 | R@3 | MRR | nDCG@3 | MAP@3 | False-alarm rate |
|---|---|---|---|---|---|---|
| A  Original (whole-doc TF-IDF) | 0.32 | 0.85 | 0.81 | 0.81 | 0.79 | 0% |
| B  BM25, whole document | 0.33 | 0.85 | 0.85 | 0.84 | 0.83 | 0% |
| C  TF-IDF, passages | 0.36 | 0.92 | 0.91 | 0.90 | 0.89 | 0% |
| D  BM25, passages | 0.33 | 0.85 | 0.87 | 0.85 | 0.85 | 0% |
| E  BM25, passages + expansion (shipped) | 0.40 | 1.00 | 0.98 | 0.98 | 0.98 | 0% |
| F  TF-IDF, passages + expansion | 0.40 | 1.00 | 0.98 | 0.98 | 0.98 | 0% |

MRR split by query type:

| System | System-generated queries | Natural-language queries |
|---|---|---|
| A  Original (whole-doc TF-IDF) | 0.80 | 0.81 |
| B  BM25, whole document | 1.00 | 0.82 |
| C  TF-IDF, passages | 1.00 | 0.89 |
| D  BM25, passages | 1.00 | 0.84 |
| E  BM25, passages + expansion (shipped) | 1.00 | 0.97 |
| F  TF-IDF, passages + expansion | 1.00 | 0.97 |

## Queries the shipped system still gets wrong or only partly right

- `pipe burst near the school and water is flooding the street`: expected ['leak_detection_procedure.txt'], got ['emergency_water_supply.txt', 'public_notification_boil_water.txt', 'leak_detection_procedure.txt']

## How to read these numbers

- **Development queries** (34) were used while building the module: to find bugs, and to choose the one setting
  that mattered (`expansion_weight`, see `eval/tune.py`). Their scores are optimistic.
- **Held-out queries** (14) were written and labelled *before* tuning and were never used to choose a setting.
  They were run after the engine was finished, and again after two final bug fixes (scores identical).
  They are the fairer estimate of how the module behaves on unseen questions.
- The corpus is small (11 sample documents) and the sets are small: differences of a few points are not significant.
- Queries, documents and the domain thesaurus were written by the same team, so expansion gains are probably
  optimistic. Replace the sample documents and add real user queries before quoting these numbers externally.
- Relevance is judged at document level, not passage level.
