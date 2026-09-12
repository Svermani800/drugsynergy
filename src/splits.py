"""Drug exclusion must check BOTH drug columns, not group on one column."""
import numpy as np
from sklearn.model_selection import train_test_split

def drugs(df):
    return set(df.drug_a) | set(df.drug_b)

def random_split(df, test_size=0.2, seed=42):
    # Caller supplies one row per unordered pair/cell, so repeats cannot straddle.
    if df.duplicated(["drug_a","drug_b","cell_line"]).any():
        raise ValueError("Aggregate repeated pair/cell observations before splitting")
    a,b=train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed)
    return df.iloc[a].copy(),df.iloc[b].copy()

def unseen_drug_split(df, held_out):
    held = {str(d).strip().upper() for d in held_out}
    if not held or not held <= drugs(df): raise ValueError("Unknown or empty held-out drug set")
    mask = df.drug_a.isin(held) | df.drug_b.isin(held)
    train, test = df.loc[~mask].copy(), df.loc[mask].copy()
    if train.empty or test.empty: raise ValueError("Split produced an empty partition")
    assert not (drugs(train) & held), "Held-out drug leaked into training"
    seen=drugs(train)
    # For a group holdout, distinguish one unseen from two unseen drugs.
    test["n_unseen_drugs"] = (~test.drug_a.isin(seen)).astype(int)+(~test.drug_b.isin(seen)).astype(int)
    test["cell_seen"] = test.cell_line.isin(set(train.cell_line))
    return train,test
