"""Clean scored combination summaries; never use outcome-derived predictors."""
import numpy as np
import pandas as pd

ALIASES = {"Drug1":"drug_a", "Drug2":"drug_b", "Cell line":"cell_line",
           "drug_row":"drug_a", "drug_col":"drug_b", "cell_line_name":"cell_line"}
KEY = ["drug_a", "drug_b", "cell_line"]

def load_clean(path, target="Bliss"):
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = raw.columns.str.strip()
    renamed = raw.rename(columns=ALIASES)
    needed = KEY + [target]
    missing = set(needed) - set(renamed.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if renamed.columns.duplicated().any(): raise ValueError("Ambiguous column aliases")
    df = renamed[needed].copy().rename(columns={target:"synergy"})
    for c in KEY:
        df[c] = df[c].astype("string").str.strip().str.upper().replace("", pd.NA)
    df["synergy"] = pd.to_numeric(df.synergy, errors="coerce").replace([np.inf,-np.inf], np.nan)
    invalid = df.isna().any(axis=1)
    report = {"input_rows":len(raw), "target":target,
              "missing_by_column":raw.isna().sum().to_dict(), "invalid_rows":int(invalid.sum())}
    df = df.loc[~invalid].copy()
    same = df.drug_a.eq(df.drug_b)
    report["self_combination_rows_removed"] = int(same.sum())
    df = df.loc[~same].copy()
    pairs = np.sort(df[["drug_a","drug_b"]].to_numpy(dtype=str), axis=1)
    df["drug_a"], df["drug_b"] = pairs[:,0], pairs[:,1]
    report["valid_measurement_rows"] = len(df)
    report["repeated_pair_cell_rows"] = int(df.duplicated(KEY).sum())
    # A first-version target is the mean across all measurements of this pair/cell.
    # No study metadata is available; do not claim these are all true replicates.
    clean = df.groupby(KEY, as_index=False).agg(synergy=("synergy","mean"),
                  n_measurements=("synergy","size"), synergy_sd=("synergy","std"))
    if clean.empty: raise ValueError("No valid combinations remain")
    report.update(clean_rows=len(clean), drugs=len(set(clean.drug_a)|set(clean.drug_b)),
                  pairs=len(clean[["drug_a","drug_b"]].drop_duplicates()),
                  cell_lines=clean.cell_line.nunique(),
                  target_summary=clean.synergy.describe().to_dict())
    return clean, report
