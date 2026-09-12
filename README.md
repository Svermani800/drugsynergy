# Drug combination synergy: unseen-drug baseline

A beginner-friendly first version of a cancer drug-combination regression project. The main question is whether a model can predict synergy for a drug absent from its training data.

This version loads `drugcombs_scored.csv`, audits and cleans the data, runs a mean predictor and regularized linear regression, and compares a random split with leave-one-drug-out evaluation. It does not yet test whether molecular or genomic features improve generalization.

## Dataset and target

The supplied CSV contains `ID, Drug1, Drug2, Cell line, ZIP, Bliss, Loewe, HSA`. Use **Bliss** as the continuous target. A positive score indicates greater effect than the Bliss independence expectation, but we do not assign clinical labels or probability/confidence estimates.

For public data, [DrugComb's versioned summary table](https://zenodo.org/records/11102665) is a suitable starting point because it provides existing synergy summaries. See the [original DrugComb paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6602441/) for the database and scoring context. The supplied file's exact release, scoring pipeline, and study provenance have not been established; similar column contents do not establish provenance. This run uses the supplied file, not the downloaded public reference table.

| Column | Role |
| --- | --- |
| Drug1 / drug_a | Drug identity |
| Drug2 / drug_b | Drug identity |
| Cell line / cell_line | Cell identity |
| Bliss / synergy | Regression target |
| ID | Source identifier; not a predictor |
| ZIP, Loewe, HSA | Alternative outcomes; never predictors of Bliss |
| n_measurements, synergy_sd | Audit summaries only; never predictors |

Doses, viability, fingerprints, and genomic data are absent from this file. Do not invent them. Rows are treated as precomputed combination summaries, not individual dose measurements.

## Run

Use Python 3.11 or newer (the recorded run used Python 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Copy your supplied CSV into data/raw/drugcombs_scored.csv first.
python -m unittest discover -s tests -v
python -m src.run --data data/raw/drugcombs_scored.csv --max-drugs 10
```

The default evaluates the ten drugs with the most cleaned observations, selected using counts rather than outcomes. This is a first benchmark on well-covered drugs, not a representative estimate for all drugs. Exact selected names and options are in `results/run_config.json`. For one specified drug:

```bash
python -m src.run --data data/raw/drugcombs_scored.csv --held-out 5-FU --out results_5fu
```

For all drugs with at least 20 cleaned test observations (potentially slow and disk intensive because each fold writes its partitions):

```bash
python -m src.run --data data/raw/drugcombs_scored.csv --max-drugs 0 --min-test 20 --out results_all
```

Use a distinct output directory for a different experiment to avoid retaining obsolete fold files. `data/processed/clean.csv` is regenerated each run. `requirements-lock.txt` records the actual environment; `requirements.txt` allows compatible versions. To open learning notebooks, optionally install Jupyter and choose this environment's kernel.

## Cleaning decisions

1. Require the three identity columns and the selected target.
2. Trim whitespace and uppercase identities. This merges case variants but does **not** resolve chemical synonyms, salt forms, or misspellings.
3. Reject missing identities and nonnumeric/nonfinite targets; remove same-drug combinations.
4. Sort each drug pair so A+B and B+A share the same key.
5. Average all valid measurements for each unordered pair and cell line. Store their count and standard deviation for auditing.

This targets average synergy for a pair/cell, weighting each resulting row equally. Repeats are not necessarily biological replicates: the file lacks study and dose-range metadata. Source-level harmonization should precede a scientific conclusion. Do not average across studies in a later study-holdout evaluation.

Aggregation occurs before the split because all measurements of the same pair/cell are intentionally one observation. No training observation shares those measurements with a test observation. Distinct cell lines for the same drug pair can occur on both sides of the random split; that evaluation is not an unseen-pair test.

## How the split prevents leakage

`src/splits.py` places every row with the held-out drug in **either** drug column into test. It asserts that the selected drug is absent from both training columns. Grouping on `Drug1` alone would fail when that drug appears as `Drug2`.

`unseen_drug_split(df, ["DRUG X", "DRUG Y"])` also supports a group of held-out drugs. Test rows are tagged with the actual number of drugs absent from training (one or two), and whether the cell line was seen. Metrics for one- and two-unseen strata are saved separately when present. Even single-drug holdout can incidentally remove every observation of a partner drug, so check these flags.

Every model and encoder is fitted afresh on training data. No target-derived score is an input. Ridge alpha is fixed at 10; there is no test-set tuning. For later tuning, make inner held-out-drug splits using only the outer training data and fit every learned preprocessing step inside them. Resolve drug aliases before claiming generalization to a chemically unseen compound; this version guarantees exclusion of normalized **names**, not chemical structures.

## Baselines explained

- **Mean:** predicts the training target average for every row.
- **Ridge identity:** regularized linear regression with shared one-hot drug features plus cell-line one-hot features. Adding the two drug vectors makes predictions invariant to drug order.

An unseen drug has an all-zero identity vector, so its prediction comes from the known partner, cell line, and intercept. That is the intended limitation of this baseline. It has no information about a new drug's chemistry. Unknown cell lines also get zero identity features.

## Outputs and interpretation

`results/eda.json` contains dataset counts, missingness, target distribution, and input SHA-256. `extreme_scores.csv` lists aggregated observations with absolute scores above 100 as an investigation aid. This is a flag, not a proven invalidity rule: **no score clipping or outlier removal is performed**. Both a full histogram and a central-range view are generated.

`metrics.csv` reports RMSE, MAE, R², Pearson, and Spearman by fold/model. Correlations are undefined for constant predictors and appear blank. `comparison.csv` contains the random result and the unweighted mean of per-drug fold metrics. LODO folds overlap, so this is not an independent pooled test set, and the mean fold RMSE is not pooled RMSE. Inspect individual held-out drugs rather than relying on the mean alone. Random and held-out evaluations also differ in test composition and training size.

Predictions, split membership, and `split_audit.json` are saved locally for review. Models saved by the random evaluation were trained only on its training partition; they are demonstration artifacts, not final deployment models. Raw data, processed rows, per-row predictions, split membership, and model binaries are excluded from Git.

See `results/FINDINGS.md` for the actual run. Extreme target values must be investigated before model comparisons can support conclusions. Calibration, prediction intervals, classification thresholds, SHAP, and molecular/genomic models are future work; a scatter plot is not uncertainty calibration.

## Batch prediction

Create a CSV containing `drug_a,drug_b,cell_line`, then run:

```bash
python -m src.predict results/ridge_identity.joblib candidates.csv ranked_predictions.csv
```

The output is sorted by predicted Bliss and includes unseen-drug and cell-coverage flags. Use models from the default Bliss run. No validated uncertainty estimate is available in this version.

## Code map

- `src/data.py`: validate, normalize, aggregate, audit.
- `src/features.py`: symmetric train-fitted identity representation.
- `src/splits.py`: reproducible random and drug-exclusion splits.
- `src/models.py`: mean and Ridge baselines.
- `src/evaluate.py`: regression and rank-correlation metrics.
- `src/run.py`: EDA, training, predictions, and saved audits.
- `src/predict.py`: batch predictions.
- `notebooks/`: three short walkthroughs of these modules.
- `tests/`: exclusion in both columns, group holdouts, unknown handling, pair symmetry, aggregation, and metric edge cases.

## Next experiment

First trace extreme scores to source experiments and document score units and calculation failures. Then map normalized names to stable chemical identifiers and SMILES, add fingerprints and descriptors with symmetric pair features, and compare them against this baseline on the same outer folds. Add harmonized cell-line expression later, with train-only scaling and dimensionality reduction. Compare drug-only, cell-only, and combined features under identical splits and report variation across held-out drugs.
