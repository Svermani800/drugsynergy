"""Curate human cancer cell lines and attach stable model identifiers."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def normalize_name(value: object) -> str:
    """Normalize identifiers for exact alias matching, never fuzzy matching."""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def build_cell_line_map(metadata_path: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Return unambiguous alias -> model ID and curated model metadata."""
    metadata = pd.read_csv(metadata_path, low_memory=False)
    aliases: dict[str, set[str]] = defaultdict(set)
    records = []
    for row in metadata.itertuples(index=False):
        species = str(getattr(row, "species", ""))
        model_type = str(getattr(row, "model_type", ""))
        cancer_type = str(getattr(row, "cancer_type", ""))
        if species.lower() not in {"human", "homo sapiens"} or model_type.lower() != "cell line":
            continue
        if not cancer_type or cancer_type.lower() == "nan":
            continue
        broad_id = str(getattr(row, "BROAD_ID", ""))
        model_id = broad_id if broad_id and broad_id.lower() != "nan" else str(row.model_id)
        candidate_names = [row.model_name, getattr(row, "CCLE_ID", "")]
        candidate_names += str(getattr(row, "synonyms", "")).split(";")
        for name in candidate_names:
            key = normalize_name(name)
            if key and key != "NAN":
                aliases[key].add(model_id)
        records.append(
            {
                "model_id": model_id,
                "sanger_model_id": str(row.model_id),
                "model_name": row.model_name,
                "tissue": getattr(row, "tissue", None),
                "cancer_type": cancer_type,
                "cancer_type_detail": getattr(row, "cancer_type_detail", None),
                "tissue_status": getattr(row, "tissue_status", None),
                "ccle_id": getattr(row, "CCLE_ID", None),
                "cosmic_id": getattr(row, "COSMIC_ID", None),
                "expression_available": bool(getattr(row, "expression_data", False)),
            }
        )
    unique = {alias: next(iter(ids)) for alias, ids in aliases.items() if len(ids) == 1}
    return unique, pd.DataFrame(records).drop_duplicates("model_id")


def curate_cancer_rows(
    combinations: pd.DataFrame, metadata_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Keep rows with an unambiguous match to a human cancer cell-line model."""
    alias_map, metadata = build_cell_line_map(metadata_path)
    curated = combinations.copy()
    curated["cell_key"] = curated.cell_line.map(normalize_name)
    curated["model_id"] = curated.cell_key.map(alias_map)
    matched = curated.model_id.notna()
    curated = curated.loc[matched].merge(metadata, on="model_id", how="left", validate="many_to_one")
    report = {
        "input_rows": int(len(combinations)),
        "input_cell_lines": int(combinations.cell_line.nunique()),
        "matched_rows": int(len(curated)),
        "matched_cell_lines": int(curated.cell_line.nunique()),
        "unmatched_cell_lines": sorted(combinations.loc[~combinations.cell_line.isin(curated.cell_line), "cell_line"].unique()),
        "match_method": "Exact normalized match against unique model name, CCLE ID, or curated synonym",
    }
    return curated, metadata, report
