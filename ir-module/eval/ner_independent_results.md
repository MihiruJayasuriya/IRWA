# NER: independent evaluation

Reproduce with `python eval/ner_independent_eval.py`. Labels were written before `ner.py` was read or run, and are broader than its scope, so this shows both its accuracy where it applies and where its coverage stops. Small sets: treat differences of a few points as noise. Types the extractor implements but these labels do not cover (WATER_QUALITY, PRESSURE, CONCENTRATION) are shown for information and left out of the precision figures. Strict recall is lower than lenient recall where span conventions differ (for example the extractor includes "per day" in a VOLUME span).

## Development set

31 texts. Same type + overlapping span counts as correct.

| Type | Gold | Precision | Recall | F1 | Strict recall |
|---|---|---|---|---|---|
| DROUGHT_STAGE | 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| DURATION | 11 | 1.00 | 0.91 | 0.95 | 0.91 |
| FACILITY | 17 | 1.00 | 0.53 | 0.69 | 0.53 |
| PERCENTAGE | 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| VOLUME | 2 | 1.00 | 1.00 | 1.00 | 0.00 |
| ZONE | 10 | 0.91 | 1.00 | 0.95 | 0.90 |
| DATE_REF (not implemented) | 8 | n/a | 0.00 | n/a | 0.00 |
| INCIDENT (not implemented) | 17 | n/a | 0.00 | n/a | 0.00 |
| WATER_QUALITY (implemented, but not labelled in this set) | 0 | 0.00 | n/a | n/a | n/a |

**Implemented types:** precision 0.97, recall 0.81, F1 0.89 (39 correct, 1 extra, 9 missed).
**All labelled types:** 39 of 73 labelled entities recovered (53%).

Errors on implemented types:

- missed `pipe` (FACILITY) in: There is a pipe leak in Zone 4 and water is flooding the road.
- missed `tap` (FACILITY) in: Brown water is coming out of the tap in the West zone.
- missed `sensor` (FACILITY) in: The reservoir sensor has shown 42.4% for 10 days.
- missed `a month` (DURATION) in: Meters are read once a month.
- missed `tanker` (FACILITY) in: Please send a tanker to zone 2 tonight.
- missed `water main` (FACILITY) in: The burst water main flooded three streets last night.
- missed `pipe` (FACILITY) in: Turn the valve, then check the pipe for a leak.
- extra `west` (ZONE) in: The wind is blowing from the west, about 50 meters from the office.
- missed `dam` (FACILITY) in: The dam is at 85% capacity.
- missed `pipeline` (FACILITY) in: A leak in the pipeline caused an outage of 3 days in Region 5.

## Held-out set

20 texts. Same type + overlapping span counts as correct.

| Type | Gold | Precision | Recall | F1 | Strict recall |
|---|---|---|---|---|---|
| DROUGHT_STAGE | 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| DURATION | 5 | 1.00 | 1.00 | 1.00 | 1.00 |
| FACILITY | 10 | 0.83 | 0.50 | 0.62 | 0.50 |
| PERCENTAGE | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| VOLUME | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| ZONE | 7 | 0.88 | 1.00 | 0.93 | 0.71 |
| DATE_REF (not implemented) | 6 | n/a | 0.00 | n/a | 0.00 |
| INCIDENT (not implemented) | 11 | n/a | 0.00 | n/a | 0.00 |

**Implemented types:** precision 0.93, recall 0.83, F1 0.88 (25 correct, 2 extra, 5 missed).
**All labelled types:** 25 of 47 labelled entities recovered (53%).

Errors on implemented types:

- missed `pipe` (FACILITY) in: There is a burst pipe outside my house in Zone 1.
- missed `Water main` (FACILITY) in: Water main repairs will take about 8 hours, starting on Wednesday.
- missed `tanker` (FACILITY) in: Send the tanker to North zone before tomorrow.
- missed `Sensor` (FACILITY) in: Sensor drift was found at the treatment facility last week.
- extra `west` (ZONE) in: He drove west for 30 minutes.
- extra `meter` (FACILITY) in: Please meter the flow every 15 minutes.
- missed `pipeline` (FACILITY) in: The pipeline inspection is done every 12 months.
