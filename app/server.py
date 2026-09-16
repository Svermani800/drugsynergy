"""Local research interface for the validated drug-synergy model."""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.production import predict_one


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "app" / "artifacts" / "synergy_model.joblib"
SUMMARY = ROOT / "results" / "production" / "summary.json"

app = Flask(__name__)
model_bundle = joblib.load(ARTIFACT) if ARTIFACT.exists() else None
validation_summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/metadata")
def metadata():
    if model_bundle is None:
        return jsonify({"ready": False, "message": "Train the production model first."}), 503
    return jsonify(
        {
            "ready": True,
            "drugs": model_bundle["drug_names"],
            "cell_lines": model_bundle["cell_lines"],
            "model": model_bundle["winner"],
            "validation": validation_summary.get("locked_audit_metrics", {}),
            "development": validation_summary.get("development_metrics", {}),
            "data": {
                "rows": validation_summary.get("final_rows"),
                "drugs": len(model_bundle["drug_names"]),
                "cell_lines": len(model_bundle["cell_lines"]),
                "expression_release": validation_summary.get("expression_release"),
            },
            "audit_drugs": validation_summary.get("locked_audit_drug_names", []),
            "audit_protocol": validation_summary.get("audit_protocol"),
        }
    )


@app.post("/api/predict")
def predict():
    if model_bundle is None:
        return jsonify({"error": "Model artifact is unavailable."}), 503
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_one(model_bundle, payload["drug_a"], payload["drug_b"], payload["cell_line"])
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(result)


@app.post("/api/batch")
def batch_predict():
    if model_bundle is None:
        return jsonify({"error": "Model artifact is unavailable."}), 503
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "Choose a CSV file."}), 400
    try:
        frame = pd.read_csv(uploaded)
    except Exception:
        return jsonify({"error": "The uploaded file is not a readable CSV."}), 400
    required = ["drug_a", "drug_b", "cell_line"]
    missing = [column for column in required if column not in frame]
    if missing:
        return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400
    if len(frame) > 5000:
        return jsonify({"error": "Local batch uploads are limited to 5,000 rows."}), 400
    output = []
    for row in frame[required].itertuples(index=False):
        record = {
            "drug_a": row.drug_a,
            "drug_b": row.drug_b,
            "cell_line": row.cell_line,
            "predicted_bliss": float("nan"),
        }
        try:
            record.update(predict_one(model_bundle, str(row.drug_a), str(row.drug_b), str(row.cell_line)))
            record["error"] = ""
        except ValueError as error:
            record["error"] = str(error)
        output.append(record)
    result = pd.DataFrame(output).sort_values("predicted_bliss", ascending=False, na_position="last")
    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    return app.response_class(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=synergy_predictions.csv"},
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_ready": model_bundle is not None})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
