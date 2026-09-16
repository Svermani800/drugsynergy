"""Train and serve the curated multimodal Bliss model."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .chemical_benchmark import structure_features
from .curation import build_cell_line_map, curate_cancer_rows, normalize_name
from .data import load_clean
from .splits import drugs, unseen_drug_split


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "production"
ARTIFACTS = ROOT / "app" / "artifacts"


def load_expression_subset(path: str | Path, model_ids: set[str]) -> pd.DataFrame:
    """Read only needed model rows from a wide DepMap expression matrix."""
    selected = []
    for chunk in pd.read_csv(path, chunksize=64, low_memory=False):
        id_column = "ModelID" if "ModelID" in chunk.columns else chunk.columns[0]
        keep = chunk[id_column].astype(str).isin(model_ids)
        if keep.any():
            selected.append(chunk.loc[keep].rename(columns={id_column: "model_id"}))
    if not selected:
        raise ValueError("No curated cell lines matched the expression matrix")
    expression = pd.concat(selected, ignore_index=True).drop_duplicates("model_id")
    numeric = expression.drop(columns="model_id").apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().mean().ge(0.95)]
    numeric = numeric.fillna(numeric.median())
    return pd.concat([expression[["model_id"]].reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)


def prepare_cohort() -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, dict]:
    clean, input_report = load_clean(RAW / "drugcombs_scored.csv")
    cancer, metadata, cell_report = curate_cancer_rows(clean, RAW / "model_list_latest.csv.gz")
    name_to_structure, vectors, structures = structure_features(pd.read_csv(PROCESSED / "drug_structures.csv"))
    eligible = cancer.drug_a.isin(name_to_structure) & cancer.drug_b.isin(name_to_structure)
    cohort = cancer.loc[eligible].copy()
    cohort["drug_a_name"] = cohort.drug_a
    cohort["drug_b_name"] = cohort.drug_b
    cohort["drug_a"] = cohort.drug_a.map(name_to_structure)
    cohort["drug_b"] = cohort.drug_b.map(name_to_structure)
    pair = np.sort(cohort[["drug_a", "drug_b"]].to_numpy(), axis=1)
    cohort[["drug_a", "drug_b"]] = pair
    cohort = cohort.loc[cohort.drug_a.ne(cohort.drug_b)].copy()
    # Different salts or aliases can resolve to the same chemical structure.
    # Re-aggregate those rows so no biological observation receives extra weight.
    cohort["weighted_synergy"] = cohort.synergy * cohort.n_measurements
    group_columns = ["drug_a", "drug_b", "cell_line", "model_id", "tissue", "cancer_type"]
    cohort = (
        cohort.groupby(group_columns, as_index=False, dropna=False)
        .agg(weighted_synergy=("weighted_synergy", "sum"), n_measurements=("n_measurements", "sum"))
    )
    cohort["synergy"] = cohort.weighted_synergy / cohort.n_measurements
    cohort = cohort.drop(columns="weighted_synergy")
    report = {
        "input": input_report,
        "cell_line_curation": cell_report,
        "eligible_rows": int(len(cohort)),
        "eligible_drug_names": int(len(name_to_structure)),
        "unique_structures": int(len(vectors)),
        "eligible_cell_lines": int(cohort.cell_line.nunique()),
        "extreme_targets_abs_gt_100": int(cohort.synergy.abs().gt(100).sum()),
    }
    return cohort, vectors, structures, report


@dataclass
class FeatureBuilder:
    drug_vectors: dict[str, np.ndarray]
    expression: pd.DataFrame
    n_genes: int = 96
    n_expression_components: int = 20

    def fit(self, frame: pd.DataFrame):
        available = self.expression.set_index("model_id")
        train_ids = sorted(set(frame.model_id) & set(available.index))
        if len(train_ids) < 3:
            raise ValueError("Too few expression-matched cell lines in training")
        expression = available.loc[train_ids]
        variances = expression.var(axis=0).sort_values(ascending=False)
        self.genes_ = variances.head(min(self.n_genes, len(variances))).index.tolist()
        self.expression_scaler_ = StandardScaler().fit(expression[self.genes_])
        scaled = self.expression_scaler_.transform(expression[self.genes_])
        components = min(self.n_expression_components, len(train_ids) - 1, len(self.genes_))
        self.expression_pca_ = PCA(n_components=components, random_state=42).fit(scaled)
        self.expression_lookup_ = {
            model_id: self.expression_pca_.transform(
                self.expression_scaler_.transform(available.loc[[model_id], self.genes_])
            )[0].astype(np.float32)
            for model_id in available.index
        }
        chemical = self._chemical(frame)
        self.chemical_scaler_ = StandardScaler(with_mean=False).fit(chemical)
        self.context_encoder_ = OneHotEncoder(handle_unknown="ignore").fit(frame[["tissue", "cancer_type"]].fillna("Unknown"))
        return self

    def _chemical(self, frame: pd.DataFrame) -> csr_matrix:
        a = np.stack([self.drug_vectors[x] for x in frame.drug_a])
        b = np.stack([self.drug_vectors[x] for x in frame.drug_b])
        return csr_matrix(np.c_[a + b, np.abs(a - b)], dtype=np.float32)

    def transform(self, frame: pd.DataFrame, multimodal: bool = True) -> csr_matrix:
        chemical = self.chemical_scaler_.transform(self._chemical(frame))
        if not multimodal:
            return chemical
        missing = set(frame.model_id) - set(self.expression_lookup_)
        if missing:
            raise ValueError(f"Missing expression data for {len(missing)} model IDs")
        biological = csr_matrix(np.stack([self.expression_lookup_[x] for x in frame.model_id]))
        context = self.context_encoder_.transform(frame[["tissue", "cancer_type"]].fillna("Unknown"))
        return hstack([chemical, biological, context], format="csr")


def xgb_model(depth: int = 4, estimators: int = 220, objective: str = "reg:pseudohubererror") -> XGBRegressor:
    return XGBRegressor(
        objective=objective,
        n_estimators=estimators,
        max_depth=depth,
        learning_rate=0.04,
        min_child_weight=12,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )


def score(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
        "spearman": float(pd.Series(y).corr(pd.Series(prediction), method="spearman")),
    }


def evaluate_candidate(name: str, train: pd.DataFrame, test: pd.DataFrame, expression: pd.DataFrame, vectors: dict):
    target = train.synergy.clip(-100, 100)
    if name == "mean_baseline":
        prediction = np.full(len(test), target.mean())
        train_sample = train.sample(min(30_000, len(train)), random_state=42)
        train_prediction = np.full(len(train_sample), target.mean())
        result = score(test.synergy.to_numpy(), prediction)
        result.update({f"train_{key}": value for key, value in score(train_sample.synergy.to_numpy(), train_prediction).items()})
        return result, prediction
    features = FeatureBuilder(vectors, expression).fit(train)
    multimodal = name != "xgb_chemical"
    x_train = features.transform(train, multimodal=multimodal)
    x_test = features.transform(test, multimodal=multimodal)
    if name == "ridge_multimodal":
        model = Ridge(alpha=20.0, solver="lsqr")
    elif name == "xgb_multimodal_depth3":
        model = xgb_model(depth=3)
    elif name == "xgb_multimodal_depth5":
        model = xgb_model(depth=5, estimators=350)
    elif name == "xgb_multimodal_square_depth4":
        model = xgb_model(depth=4, estimators=350, objective="reg:squarederror")
    elif name == "xgb_multimodal_square_depth5":
        model = xgb_model(depth=5, estimators=350, objective="reg:squarederror")
    else:
        model = xgb_model(depth=4)
    model.fit(x_train, target)
    prediction = model.predict(x_test)
    train_sample = train.sample(min(30_000, len(train)), random_state=42)
    train_prediction = model.predict(features.transform(train_sample, multimodal=multimodal))
    result = score(test.synergy.to_numpy(), prediction)
    result.update({f"train_{key}": value for key, value in score(train_sample.synergy.to_numpy(), train_prediction).items()})
    return result, prediction


def benchmark_and_train() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cohort, vectors, structures, report = prepare_cohort()
    expression = load_expression_subset(RAW / "OmicsExpressionProteinCodingGenesTPMLogp1_24Q4.csv", set(cohort.model_id))
    cohort = cohort.loc[cohort.model_id.isin(expression.model_id)].copy()
    counts = pd.concat([cohort.drug_a, cohort.drug_b]).value_counts()
    held_out = counts.head(13).index.tolist()
    development = held_out[:7]
    pilot_audit = held_out[7:10]
    locked_audit = held_out[10:13]
    candidates = [
        "mean_baseline",
        "ridge_multimodal",
        "xgb_chemical",
        "xgb_multimodal_depth3",
        "xgb_multimodal_depth4",
        "xgb_multimodal_depth5",
        "xgb_multimodal_square_depth4",
        "xgb_multimodal_square_depth5",
    ]
    rows = []
    predictions = []
    for held in development:
        train, test = unseen_drug_split(cohort, [held])
        for candidate in candidates:
            metrics, pred = evaluate_candidate(candidate, train, test, expression, vectors)
            rows.append({"stage": "development", "held_out": held, "model": candidate, "n_test": len(test), **metrics})
            predictions.extend(
                {"stage": "development", "held_out": held, "model": candidate, "observed": float(y), "predicted": float(p)}
                for y, p in zip(test.synergy, pred)
            )
    development_results = pd.DataFrame(rows)
    winner = development_results.groupby("model").rmse.mean().sort_values().index[0]
    for held in locked_audit:
        train, test = unseen_drug_split(cohort, [held])
        metrics, pred = evaluate_candidate(winner, train, test, expression, vectors)
        rows.append({"stage": "locked_audit", "held_out": held, "model": winner, "n_test": len(test), **metrics})
        predictions.extend(
            {"stage": "locked_audit", "held_out": held, "model": winner, "observed": float(y), "predicted": float(p)}
            for y, p in zip(test.synergy, pred)
        )
    results = pd.DataFrame(rows)
    residuals = pd.DataFrame(predictions)
    results.to_csv(RESULTS / "validation_metrics.csv", index=False)
    residuals.to_csv(RESULTS / "validation_predictions.csv.gz", index=False)
    audit_residual = residuals.loc[residuals.stage.eq("locked_audit"), "observed"] - residuals.loc[residuals.stage.eq("locked_audit"), "predicted"]
    abs_error = audit_residual.abs()
    error_quantiles = {str(q): float(abs_error.quantile(q)) for q in (0.5, 0.8, 0.9, 0.95)}
    final_features = FeatureBuilder(vectors, expression).fit(cohort)
    multimodal = winner != "xgb_chemical"
    final_x = final_features.transform(cohort, multimodal=multimodal)
    if winner == "mean_baseline":
        final_model = DummyRegressor(strategy="mean")
    elif winner == "ridge_multimodal":
        final_model = Ridge(alpha=20.0, solver="lsqr")
    elif winner == "xgb_multimodal_depth3":
        final_model = xgb_model(depth=3)
    elif winner == "xgb_multimodal_depth5":
        final_model = xgb_model(depth=5, estimators=350)
    elif winner == "xgb_multimodal_square_depth4":
        final_model = xgb_model(depth=4, estimators=350, objective="reg:squarederror")
    elif winner == "xgb_multimodal_square_depth5":
        final_model = xgb_model(depth=5, estimators=350, objective="reg:squarederror")
    else:
        final_model = xgb_model(depth=4)
    final_model.fit(final_x, cohort.synergy.clip(-100, 100))
    aliases, metadata = build_cell_line_map(RAW / "model_list_latest.csv.gz")
    name_to_structure = dict(zip(structures.drug, structures.structure_id))
    structure_to_name = structures.groupby("structure_id").drug.first().to_dict()
    artifact = {
        "model": final_model,
        "features": final_features,
        "multimodal": multimodal,
        "winner": winner,
        "drug_to_structure": name_to_structure,
        "cell_alias_to_model": aliases,
        "cell_metadata": metadata.set_index("model_id").to_dict("index"),
        "drug_names": sorted(name_to_structure),
        "cell_lines": sorted(cohort.cell_line.unique()),
        "error_quantiles": error_quantiles,
        "thresholds": {"antagonistic": -10.0, "synergistic": 10.0},
    }
    joblib.dump(artifact, ARTIFACTS / "synergy_model.joblib", compress=3)
    development_table = development_results.groupby("model")[["rmse", "mae", "r2", "spearman"]].mean()
    development_summary = development_table.astype(object).where(development_table.notna(), None).to_dict("index")
    summary = {
        **report,
        "expression_release": "DepMap Public 24Q4",
        "expression_rows": int(len(expression)),
        "expression_genes_available": int(expression.shape[1] - 1),
        "final_rows": int(len(cohort)),
        "development_drugs": development,
        "development_drug_names": [structure_to_name[x] for x in development],
        "pilot_audit_drugs": pilot_audit,
        "pilot_audit_drug_names": [structure_to_name[x] for x in pilot_audit],
        "locked_audit_drugs": locked_audit,
        "locked_audit_drug_names": [structure_to_name[x] for x in locked_audit],
        "audit_protocol": "Final audit drugs were reserved after the initial pilot audit and never used for candidate selection",
        "selected_model": winner,
        "training_target_policy": "Clip training labels to [-100, 100]; never clip validation labels",
        "development_metrics": development_summary,
        "locked_audit_metrics": results.loc[results.stage.eq("locked_audit"), ["rmse", "mae", "r2", "spearman"]].mean().to_dict(),
        "empirical_absolute_error_quantiles": error_quantiles,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def prediction_frame(artifact: dict, drug_a: str, drug_b: str, cell_line: str) -> pd.DataFrame:
    drug_a = drug_a.strip().upper()
    drug_b = drug_b.strip().upper()
    cell_line = cell_line.strip().upper()
    mapping = artifact["drug_to_structure"]
    if drug_a not in mapping or drug_b not in mapping:
        missing = [name for name in (drug_a, drug_b) if name not in mapping]
        raise ValueError(f"No validated structure mapping for: {', '.join(missing)}")
    alias = normalize_name(cell_line)
    model_id = artifact["cell_alias_to_model"].get(alias)
    if model_id is None or model_id not in artifact["cell_metadata"]:
        raise ValueError(f"No curated cancer-model mapping for cell line: {cell_line}")
    metadata = artifact["cell_metadata"][model_id]
    pair = sorted([mapping[drug_a], mapping[drug_b]])
    if pair[0] == pair[1]:
        raise ValueError("Choose two drugs with different parent chemical structures")
    return pd.DataFrame(
        [{
            "drug_a": pair[0], "drug_b": pair[1], "model_id": model_id,
            "tissue": metadata.get("tissue"), "cancer_type": metadata.get("cancer_type"),
        }]
    )


def predict_one(artifact: dict, drug_a: str, drug_b: str, cell_line: str) -> dict:
    frame = prediction_frame(artifact, drug_a, drug_b, cell_line)
    x = artifact["features"].transform(frame, multimodal=artifact["multimodal"])
    value = float(artifact["model"].predict(x)[0])
    lower = value - artifact["error_quantiles"]["0.8"]
    upper = value + artifact["error_quantiles"]["0.8"]
    thresholds = artifact["thresholds"]
    label = "synergistic" if value > thresholds["synergistic"] else "antagonistic" if value < thresholds["antagonistic"] else "additive / uncertain"
    return {
        "predicted_bliss": value,
        "interpretation": label,
        "validation_interval_80": [lower, upper],
        "tissue": frame.tissue.iloc[0],
        "cancer_type": frame.cancer_type.iloc[0],
        "model": artifact["winner"],
        "research_only": True,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark_and_train(), indent=2))
