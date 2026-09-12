# First run findings

Input: user-supplied `drugcombs_scored.csv`; target: Bliss. No clipping or outcome filtering.

- Input: 498,865 rows; 10 rows have missing required identities.
- Removed 382 same-drug observations.
- Averaged 498,473 valid measurements into 405,465 unordered pair/cell observations (93,008 repeated keys beyond the first).
- 5,350 normalized drug names, 78,751 unordered drug pairs, 123 cell lines.
- Target median 0.02; interquartile range −3.435 to 3.38.
- Target minimum −221,232.877; maximum 1,689,573.201.
- 499 aggregated scores have absolute value above 100. This is an audit flag, not automatic grounds for deletion.

## Baseline results

| Evaluation | Model | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| lodo_macro | mean | 70.444 | 69.894 | -91.280 |
| lodo_macro | ridge_identity | 786.004 | 413.382 | -7344.817 |
| random | mean | 4023.018 | 123.765 | -0.000 |
| random | ridge_identity | 4123.189 | 179.476 | -0.050 |

The ten held-out drugs were selected by observation count, independently of target scores. Each has its own newly fitted models. This is not full leave-one-drug-out across every drug. The LODO summary averages per-drug metrics; folds overlap. Extreme scores are distributed unevenly across folds, so the much smaller LODO mean RMSE does not show that unseen-drug prediction is easier. Ridge does not improve RMSE over the mean predictor here. These results do not establish the value of multimodal features.

Investigate original experiment records, numeric units, scoring failures, aliases, and study provenance before interpreting these outcomes. Scores were intentionally retained to expose this issue. No calibrated uncertainty or clinical classification has been produced.

## Held-out fold mapping

- `lodo_000`: RUXOLITINIB
- `lodo_001`: TEMOZOLOMIDE
- `lodo_002`: SORAFENIB
- `lodo_003`: SUNITINIB
- `lodo_004`: ZOLINZA
- `lodo_005`: DASATINIB
- `lodo_006`: LAPATINIB
- `lodo_007`: METHOTREXATE
- `lodo_008`: 5-FU
- `lodo_009`: BORTEZOMIB

## Validation

Eight tests passed, including exclusion from both drug columns, multiple held-out drugs, incidental double-unseen rows, pair-order invariance, train-only vocabulary, repeated-key aggregation, deterministic random splitting, and constant-prediction correlation handling. The full CSV pipeline ran successfully with ten held-out folds.
