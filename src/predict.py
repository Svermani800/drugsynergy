"""Batch prediction with explicit identity-coverage flags; no confidence claim."""
import argparse
import joblib
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("csv"); ap.add_argument("output")
    a=ap.parse_args()
    model=joblib.load(a.model)  # Only load model files you trust.
    df=pd.read_csv(a.csv)
    for c in ["drug_a","drug_b","cell_line"]:
        df[c]=df[c].astype("string").str.strip().str.upper()
        if df[c].isna().any() or df[c].eq("").any(): raise ValueError(f"Missing {c}")
    enc=model.steps[0][1]
    known=set(enc.drug_encoder_.categories_[0])
    df["n_unseen_drugs"]=(~df.drug_a.isin(known)).astype(int)+(~df.drug_b.isin(known)).astype(int)
    df["cell_seen"]=df.cell_line.isin(enc.cell_encoder_.categories_[0])
    df["predicted_bliss"]=model.predict(df)
    df.sort_values("predicted_bliss",ascending=False).to_csv(a.output,index=False)
if __name__=="__main__": main()
