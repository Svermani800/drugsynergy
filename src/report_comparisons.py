import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/drug-synergy-matplotlib')
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
p=Path('.')
for directory in ['model_comparison','chemical_comparison']:
    out=p/'results'/directory
    summary=pd.read_csv(out/'summary.csv')
    use=summary[summary.subset.eq('all')]
    parts=[]
    for evaluation in ['random','lodo_macro']:
        block=use[use.evaluation.eq(evaluation)].sort_values('rmse')
        parts.append('## '+evaluation+'\n\n| Model | RMSE | MAE | R² | Spearman |\n| --- | ---: | ---: | ---: | ---: |')
        for row in block.itertuples():parts.append(f'| {row.model} | {row.rmse:.3f} | {row.mae:.3f} | {row.r2:.3f} | {row.spearman:.3f} |')
        fig,ax=plt.subplots(figsize=(10,5));ax.barh(block.model,block.rmse);ax.invert_yaxis();ax.set(xlabel='RMSE (all unchanged test targets)',title=evaluation+' — '+directory)
        if directory=='model_comparison':ax.set_xscale('log');ax.set_xlabel('RMSE — log scale')
        fig.tight_layout();fig.savefig(out/f'{evaluation}_rmse.png',dpi=150);plt.close(fig)
    findings='# Model comparison results\n\nFixed parameters; no test-set tuning. LODO is the macro average over ten overlapping held-out-drug folds, not independent pooled test data. All reported scores below use unchanged test targets. `clip_train` caps only training targets at ±100; it is an exploratory sensitivity analysis, not validated data correction.\n\n'+'\n'.join(parts)
    if directory=='chemical_comparison':
        r=pd.read_csv(out/'metrics.csv');r=r[r.subset.eq('all') & r.split.ne('random')]
        pivot=r.pivot(index='held_out',columns='model',values='rmse')
        pivot.to_csv(out/'per_drug_rmse.csv')
        delta=pivot.xgboost_chemical_cell-pivot.xgboost_identity
        wins=int(delta.lt(0).sum())
        findings+=f'\n\nChemical+cell XGBoost beats identity XGBoost on RMSE for {wins}/10 held-out drugs with unchanged training targets. Mean paired RMSE change (chemical minus identity): {delta.mean():.3f}. Negative is better. This is descriptive; folds overlap and were selected by coverage.\n'
        findings+='\nThe matched cohort includes 97 resolved drug structures and 274,131 observations. It differs from the full-data benchmark and contains only one absolute score above 100. The two cohorts must not be compared as evidence for chemistry. Cell features are identity only. Automatic PubChem resolution is not expert curation; remaining cell labels need organism/disease verification. Molecular descriptors and Morgan fingerprints were combined symmetrically; structural aliases were grouped before the splits.\n'
        fig,ax=plt.subplots(figsize=(10,6));pivot[['mean','ridge_identity','xgboost_identity','ridge_chemical_cell','xgboost_chemical_cell']].plot.bar(ax=ax);ax.set(ylabel='RMSE',title='Unchanged-score performance by held-out drug');fig.tight_layout();fig.savefig(out/'per_drug_rmse.png',dpi=150);plt.close(fig)
        print(pivot.to_string());print('XGB chemistry wins',wins)
    else:
        findings+='\n\nFull-data XGBoost did not establish superiority: extreme scores dominate the squared-error fits. The median predictor is a necessary robust reference. Training-only clipping improves held-out errors substantially, but does not resolve source validity and cannot remove the effect of extreme test scores on RMSE.\n'
    (out/'FINDINGS.md').write_text(findings)
