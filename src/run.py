"""Run EDA and audited evaluations. Execute from the project directory."""
import argparse, hashlib, json, os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/drug-synergy-matplotlib")
from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import load_clean
from .splits import random_split, unseen_drug_split, drugs
from .models import baseline_models
from .evaluate import metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--target",default="Bliss")
    ap.add_argument("--out",default="results"); ap.add_argument("--held-out",nargs="+")
    ap.add_argument("--max-drugs",type=int,default=10,help="0 runs all eligible drugs")
    ap.add_argument("--min-test",type=int,default=20); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df,report=load_clean(a.data,a.target)
    report["source_sha256"]=hashlib.file_digest(open(a.data,"rb"),"sha256").hexdigest()
    report["aggregated_abs_score_over_100"] = int(df.synergy.abs().gt(100).sum())
    df.loc[df.synergy.abs().gt(100)].sort_values("synergy").to_csv(out/"extreme_scores.csv",index=False)
    (out/"eda.json").write_text(json.dumps(report,indent=2))
    Path("data/processed").mkdir(parents=True,exist_ok=True)
    df.to_csv("data/processed/clean.csv",index=False)
    fig,ax=plt.subplots(); ax.hist(df.synergy,bins=80); ax.set(xlabel=f"Mean {a.target} score per pair/cell",ylabel="Count")
    fig.tight_layout(); fig.savefig(out/"score_distribution.png"); plt.close(fig)
    central=df.synergy.between(-100,100)
    fig,ax=plt.subplots(); ax.hist(df.loc[central,"synergy"],bins=80)
    ax.set(xlabel=f"{a.target} score",ylabel="Count",title=f"Central range only; {(~central).sum()} observations outside view")
    fig.tight_layout(); fig.savefig(out/"score_distribution_central.png"); plt.close(fig)
    counts=pd.concat([df.drug_a,df.drug_b]).value_counts().rename_axis("drug").rename("pair_cell_count")
    counts.to_csv(out/"drug_coverage.csv")
    selected=a.held_out if a.held_out else list(counts[counts>=a.min_test].sort_values(ascending=False, kind="stable").index)
    if not a.held_out and a.max_drugs: selected=selected[:a.max_drugs]
    rows=[]; audits=[]
    def evaluate_split(label,train,test):
        audit={"split":label,"train_rows":len(train),"test_rows":len(test),
               "train_drugs":sorted(drugs(train)),"test_drugs":sorted(drugs(test)),
               "absent_from_train":sorted(drugs(test)-drugs(train))}
        audits.append(audit)
        partition=pd.concat([train.assign(partition="train"),test.assign(partition="test")])
        partition.to_csv(out/f"{label}_split.csv",index=False)
        for name,model in baseline_models().items():
            model.fit(train,train.synergy)
            pred=model.predict(test)
            rows.append({"split":label,"model":name,**metrics(test.synergy,pred)})
            predicted=test.assign(predicted_synergy=pred)
            predicted.to_csv(out/f"{label}_{name}_predictions.csv",index=False)
            if "n_unseen_drugs" in test:
                for n in (1,2):
                    mask=test.n_unseen_drugs.eq(n)
                    if mask.any(): rows.append({"split":label+f"__{n}_unseen","model":name,**metrics(test.loc[mask,"synergy"],pred[mask])})
            if label=="random":
                joblib.dump(model,out/f"{name}.joblib")
                fig,ax=plt.subplots(); ax.scatter(test.synergy,pred,s=4,alpha=.2)
                limits=[min(test.synergy.min(),pred.min()),max(test.synergy.max(),pred.max())]
                ax.plot(limits,limits,"k--"); ax.set(xlabel="Observed score",ylabel="Predicted score",title=name)
                fig.tight_layout(); fig.savefig(out/f"{name}_predicted_actual.png"); plt.close(fig)
    evaluate_split("random",*random_split(df,seed=a.seed))
    mapping={}
    for i,drug in enumerate(selected):
        label=f"lodo_{i:03d}"; mapping[label]=drug
        try: train,test=unseen_drug_split(df,[drug])
        except ValueError as e:
            audits.append({"split":label,"skipped":str(e)}); continue
        evaluate_split(label,train,test)
    results=pd.DataFrame(rows); results.to_csv(out/"metrics.csv",index=False)
    base=results[~results.split.str.contains("__")].copy()
    base["evaluation"]=base.split.map(lambda s:"random" if s=="random" else "lodo_macro")
    summary=base.groupby(["evaluation","model"])[["rmse","mae","r2","pearson","spearman"]].mean()
    summary.to_csv(out/"comparison.csv")
    summary.rmse.unstack().plot.bar(rot=0,ylabel="RMSE",title="Random vs mean held-out-drug RMSE")
    plt.tight_layout(); plt.savefig(out/"split_comparison.png"); plt.close()
    (out/"split_audit.json").write_text(json.dumps(audits,indent=2))
    (out/"run_config.json").write_text(json.dumps({**vars(a),"held_out_mapping":mapping},indent=2))
    print(json.dumps(report,indent=2)); print(summary.to_string())
if __name__=="__main__": main()
