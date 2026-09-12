"""Fixed-parameter comparison; test targets never used for fitting or tuning."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/drug-synergy-matplotlib')
import json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
from .data import load_clean
from .features import IdentityFeatures
from .splits import random_split,unseen_drug_split,drugs
from .evaluate import metrics

def models():
    xgb=dict(n_estimators=120,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,tree_method='hist',n_jobs=4,random_state=42)
    trees=dict(n_estimators=60,max_depth=14,min_samples_leaf=10,max_features='sqrt',n_jobs=4,random_state=42)
    return {'mean':(DummyRegressor(strategy='mean'),False),'median':(DummyRegressor(strategy='median'),False),
      'ridge':(Ridge(alpha=10,solver='lsqr'),False),
      'random_forest':(RandomForestRegressor(**trees),False),'extra_trees':(ExtraTreesRegressor(**trees),False),
      'xgboost':(XGBRegressor(**xgb,objective='reg:squarederror'),False),
      'ridge_clip_train':(Ridge(alpha=10,solver='lsqr'),True),
      'xgboost_clip_train':(XGBRegressor(**xgb,objective='reg:squarederror'),True)}

def main():
    out=Path('results/model_comparison');out.mkdir(parents=True,exist_ok=True)
    df,_=load_clean('data/raw/drugcombs_scored.csv')
    held=json.loads(Path('results/run_config.json').read_text())['held_out_mapping']; rows=[]; audits=[]
    config={n:{'parameters':m.get_params(),'clip_training_target':c} for n,(m,c) in models().items()}
    (out/'config.json').write_text(json.dumps({'models':config,'held_out_mapping':held,'seed':42,
      'data_sha256':hashlib.sha256(Path('data/raw/drugcombs_scored.csv').read_bytes()).hexdigest(),
      'sensitivity':'Clip training targets to [-100,100]; keep test targets unchanged. Central test subset is descriptive, not a prospective filtering rule.'},indent=2))
    for label in ['random',*held]:
        train,test=random_split(df,seed=42) if label=='random' else unseen_drug_split(df,[held[label]])
        if label!='random':assert held[label] not in drugs(train)
        enc=IdentityFeatures().fit(train); X=enc.transform(train); Z=enc.transform(test)
        audits.append({'split':label,'held_out':held.get(label),'train_rows':len(train),'test_rows':len(test),
          'unseen_names':sorted(drugs(test)-drugs(train)),'test_extremes':int(test.synergy.abs().gt(100).sum()),
          'train_extremes':int(train.synergy.abs().gt(100).sum())})
        table=test[['drug_a','drug_b','cell_line','synergy']].copy()
        for name,(model,clip) in models().items():
            start=time.time(); y=train.synergy.clip(-100,100) if clip else train.synergy
            model.fit(X,y); pred=model.predict(Z); assert np.isfinite(pred).all(); elapsed=time.time()-start
            table[name]=pred
            groups={'all':np.ones(len(test),dtype=bool),'central_abs_le_100':test.synergy.abs().le(100).to_numpy()}
            if label!='random':groups['exactly_one_unseen']=test.n_unseen_drugs.eq(1).to_numpy()
            for subset,mask in groups.items():
                if mask.any():rows.append({'split':label,'held_out':held.get(label),'model':name,'subset':subset,'fit_predict_seconds':elapsed,**metrics(test.synergy.to_numpy()[mask],pred[mask])})
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print(f'{label} {name} finished in {elapsed:.1f}s',flush=True)
        table.to_csv(out/f'{label}_predictions.csv.gz',index=False)
    (out/'split_audit.json').write_text(json.dumps(audits,indent=2))
    results=pd.DataFrame(rows);results['evaluation']=np.where(results.split.eq('random'),'random','lodo_macro')
    summary=results.groupby(['evaluation','subset','model'])[['rmse','mae','r2','pearson','spearman']].mean()
    summary.to_csv(out/'summary.csv'); print(summary.to_string(),flush=True)
if __name__=='__main__':main()
