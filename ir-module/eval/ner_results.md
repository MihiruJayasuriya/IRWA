# NER evaluation

Reproduce with `python eval/ner_eval.py`. Entity-level precision / recall / F1 on (type, normalised value) sets per text.

## Summary

| Set | Texts | Precision | Recall | F1 | How much to trust it |
|---|---|---|---|---|---|
| Development: 11 policy documents + 18 questions | 29 | 1.00 | 1.00 | 1.00 | Low: written with the extractor, so it is optimistic |
| Challenge: paraphrases and traps, **after fixes** | 30 | 1.00 | 1.00 | 1.00 | Low: the fixes were made after seeing these errors |
| Fresh: new phrasings and traps, run once after the fixes | 22 | 0.96 | 1.00 | 0.98 | **Highest**: labelled before it was run and never used to change the code |

**First run of the challenge set, before any fix** (recorded, not reproducible after the fixes): precision 0.94, recall 0.89, F1 0.91. It missed a word-number percentage (`twenty percent`), a word-number zone (`Zone Four`), a Roman-numeral stage (`Stage III`) and `mg per liter`, and wrongly read `10 meters` as a water meter and `Central Park` as a zone. All six were fixed. That 0.91 was the honest estimate before the fixes.

### Development: 11 policy documents + 18 questions

| Entity type | Precision | Recall | F1 | Correct | Wrong | Missed |
|---|---|---|---|---|---|---|
| CONCENTRATION | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| DROUGHT_STAGE | 1.00 | 1.00 | 1.00 | 6 | 0 | 0 |
| DURATION | 1.00 | 1.00 | 1.00 | 27 | 0 | 0 |
| FACILITY | 1.00 | 1.00 | 1.00 | 17 | 0 | 0 |
| PERCENTAGE | 1.00 | 1.00 | 1.00 | 11 | 0 | 0 |
| PRESSURE | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| VOLUME | 1.00 | 1.00 | 1.00 | 5 | 0 | 0 |
| WATER_QUALITY | 1.00 | 1.00 | 1.00 | 7 | 0 | 0 |
| ZONE | 1.00 | 1.00 | 1.00 | 9 | 0 | 0 |
| **All** | **1.00** | **1.00** | **1.00** | 86 | 0 | 0 |

### Challenge: paraphrases and traps, **after fixes**

| Entity type | Precision | Recall | F1 | Correct | Wrong | Missed |
|---|---|---|---|---|---|---|
| CONCENTRATION | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| DROUGHT_STAGE | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| DURATION | 1.00 | 1.00 | 1.00 | 10 | 0 | 0 |
| FACILITY | 1.00 | 1.00 | 1.00 | 6 | 0 | 0 |
| PERCENTAGE | 1.00 | 1.00 | 1.00 | 4 | 0 | 0 |
| PRESSURE | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| VOLUME | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| WATER_QUALITY | 1.00 | 1.00 | 1.00 | 4 | 0 | 0 |
| ZONE | 1.00 | 1.00 | 1.00 | 5 | 0 | 0 |
| **All** | **1.00** | **1.00** | **1.00** | 36 | 0 | 0 |

### Fresh: new phrasings and traps, run once after the fixes

| Entity type | Precision | Recall | F1 | Correct | Wrong | Missed |
|---|---|---|---|---|---|---|
| CONCENTRATION | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| DROUGHT_STAGE | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| DURATION | 1.00 | 1.00 | 1.00 | 5 | 0 | 0 |
| FACILITY | 1.00 | 1.00 | 1.00 | 4 | 0 | 0 |
| PERCENTAGE | 1.00 | 1.00 | 1.00 | 3 | 0 | 0 |
| PRESSURE | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| VOLUME | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| WATER_QUALITY | 1.00 | 1.00 | 1.00 | 3 | 0 | 0 |
| ZONE | 0.83 | 1.00 | 0.91 | 5 | 1 | 0 |
| **All** | **0.96** | **1.00** | **0.98** | 27 | 1 | 0 |

## Remaining errors

- [Fresh] `the South Pole is very cold`: wrong [('ZONE', 'South')]

## Limits

- Rule-based: it finds only the entity kinds in the schema (zones, drought stages, percentages, volumes, pressures, durations, concentrations, water-quality parameters, facilities). Clock times, frequencies (`twice weekly`) and dates are out of scope.
- **Zone names are ambiguous outside the domain**: `the South Pole` is read as the South zone.
- The gold sets are small and were annotated by the person who designed the schema. Treat every score as an illustration.

