# Pilot dataset: criteria, scenarios and responses

The standard pilot generation, without any of the design experiments:
15 criteria, the 20 scenarios from Anand's scenario pipeline (`outputs/scenarios/pilot/`), and the
responses written for them (`outputs/responses/pilot_v2/`). 76 responses in total.
Rebuild with `python pipeline/export_dataset.py` (run from `part1_data/`).

## `criteria.jsonl`
One row per criterion.

| field | meaning |
|---|---|
| `pilot_id` | 1-15; every other file refers to criteria by this number |
| `criterion_id` | id in Jash's 78-criterion shortlist, e.g. `marxism#07` |
| `tradition` | the value tradition it comes from |
| `criterion` | the criterion text ("Prefer the response that ...") |

## `scenarios.jsonl`
One row per scenario.

| field | meaning |
|---|---|
| `scenario_id` | e.g. `gen_00014` |
| `scenario` | the user message |
| `verified_criteria` | `pilot_id`s of the criteria this scenario involves: the pipeline's final, held-out-judge-verified set |
| `verified_criterion_ids` | the same criteria as `criterion_id`s |
| `responses` | list of `{response_id, kind, recommendation, text}` |

`kind` is `committed` (written to firmly recommend one option, `recommendation`; ids `<scenario_id>:r1`, `:r2`, ...)
or `natural` (the writer model's own unprompted answer, id `<scenario_id>:n`, `recommendation` is null).
Scenarios have 2-3 committed responses plus one natural response.

See `PIPELINE.md` for how the data was made, what it covers and its quality metrics.
