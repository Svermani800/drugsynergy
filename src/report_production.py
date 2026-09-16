"""Create compact validation figures for the production model card."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "production"


def main() -> None:
    metrics = pd.read_csv(RESULTS / "validation_metrics.csv")
    predictions = pd.read_csv(RESULTS / "validation_predictions.csv.gz")
    mapping = pd.read_csv(ROOT / "results" / "chemical_comparison" / "structure_mapping.csv")
    names = mapping.groupby("structure_id").drug.first().to_dict()
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text())
    for key in ("development_drugs", "pilot_audit_drugs", "locked_audit_drugs"):
        summary[key.replace("drugs", "drug_names")] = [names[x] for x in summary[key]]
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    development = metrics.loc[metrics.stage.eq("development")]
    order = development.groupby("model").rmse.mean().sort_values().index
    labels = {
        "mean_baseline": "Mean",
        "ridge_multimodal": "Ridge + all features",
        "xgb_chemical": "XGBoost + chemistry",
        "xgb_multimodal_depth3": "XGBoost + all (depth 3)",
        "xgb_multimodal_depth4": "XGBoost + all (depth 4)",
        "xgb_multimodal_depth5": "XGBoost + all (depth 5)",
        "xgb_multimodal_square_depth4": "XGBoost squared loss (depth 4)",
        "xgb_multimodal_square_depth5": "XGBoost squared loss (depth 5)",
    }
    means = development.groupby("model").rmse.mean().reindex(order)
    errors = development.groupby("model").rmse.std().reindex(order)

    audit_metrics = metrics.loc[metrics.stage.eq("locked_audit")].copy()
    audit_metrics["drug"] = audit_metrics.held_out.map(names)
    audit = predictions.loc[predictions.stage.eq("locked_audit")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.patch.set_facecolor("#f3f0e7")
    for axis in axes:
        axis.set_facecolor("#fbfaf6")
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].barh(range(len(order)), means, xerr=errors, color=["#0d685b" if x == order[0] else "#b8c4bf" for x in order])
    axes[0].set_yticks(range(len(order)), [labels[x] for x in order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE (lower is better)")
    axes[0].set_title("Development drug holdouts", loc="left")
    axes[0].grid(axis="x", alpha=.18)

    lo, hi = np.quantile(np.r_[audit.observed, audit.predicted], [.01, .99])
    bound = max(abs(lo), abs(hi), 10)
    axes[1].hexbin(audit.observed, audit.predicted, gridsize=34, mincnt=1, cmap="YlGn", extent=(-bound, bound, -bound, bound))
    axes[1].plot([-bound, bound], [-bound, bound], color="#f06f55", lw=1.5, ls="--")
    axes[1].set(xlabel="Observed Bliss", ylabel="Predicted Bliss", xlim=(-bound, bound), ylim=(-bound, bound))
    axes[1].set_title("Locked audit predictions", loc="left")
    axes[1].text(.04, .94, "central 98% shown", transform=axes[1].transAxes, va="top", fontsize=9, color="#66756f")

    audit_metrics = audit_metrics.sort_values("rmse")
    axes[2].barh(audit_metrics.drug, audit_metrics.rmse, color="#c9f05a", edgecolor="#10221d", linewidth=.8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("RMSE")
    axes[2].set_title("Three untouched audit drugs", loc="left")
    axes[2].grid(axis="x", alpha=.18)

    fig.suptitle("Synergy Atlas validation", x=.055, ha="left", fontsize=18, fontweight="bold")
    fig.text(.055, .01, "All preprocessing is fitted inside each fold. Test labels are never clipped.", fontsize=9, color="#66756f")
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(RESULTS / "validation_overview.png", dpi=180, bbox_inches="tight")
    fig.savefig(RESULTS / "validation_overview.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
