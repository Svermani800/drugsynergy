"""Structure-aware evaluation on a documented, matched chemistry cohort."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/drug-synergy-matplotlib')
import json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from rdkit import Chem
from rdkit.Chem import Descriptors,rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from .data import load_clean
from .splits import random_split,unseen_drug_split,drugs
from .features import IdentityFeatures
from .evaluate import metrics

DESCRIPTORS=[Descriptors.MolWt,Descriptors.MolLogP,Descriptors.TPSA,Descriptors.NumHDonors,Descriptors.NumHAcceptors,Descriptors.NumRotatableBonds,Descriptors.RingCount,Descriptors.FractionCSP3]

def structure_features(table):
    mappings={}; vectors={}; details=[]
    fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=256)
    for row in table.itertuples():
        if row.status!='resolved' or not isinstance(row.smiles,str):continue
        mol=Chem.MolFromSmiles(row.smiles)
        if mol is None:continue
        parent=rdMolStandardize.FragmentParent(mol)
        parent=rdMolStandardize.Uncharger().uncharge(parent)
        Chem.SanitizeMol(parent)
        # Conservative exclusion: group stereoisomers and salt forms by parent connectivity.
        canonical=Chem.MolToSmiles(parent,isomericSmiles=False)
        key=hashlib.sha256(canonical.encode()).hexdigest()[:24].upper()
        vals=np.r_[fp.GetFingerprintAsNumPy(parent),[f(parent) for f in DESCRIPTORS]].astype(np.float32)
        if not np.isfinite(vals).all():continue
        mappings[row.drug]=key; vectors[key]=vals
        details.append({'drug':row.drug,'structure_id':key,'parent_smiles':canonical,'cid':row.cid})
    return mappings,vectors,pd.DataFrame(details)

def pair_matrix(df,vectors):
    a=np.stack([vectors[x] for x in df.drug_a]); b=np.stack([vectors[x] for x in df.drug_b])
    return csr_matrix(np.c_[a+b,np.abs(a-b)])

def main():
    out=Path('results/chemical_comparison');out.mkdir(parents=True,exist_ok=True)
    full,_=load_clean('data/raw/drugcombs_scored.csv')
    mapping,vectors,details=structure_features(pd.read_csv('data/processed/drug_structures.csv'))
    details.to_csv(out/'structure_mapping.csv',index=False)
    eligible=full.drug_a.isin(mapping)&full.drug_b.isin(mapping)&~full.cell_line.isin(['3D7','HB3','DD2'])
    df=full.loc[eligible].copy()
    df.drug_a=df.drug_a.map(mapping);df.drug_b=df.drug_b.map(mapping)
    pairs=np.sort(df[['drug_a','drug_b']].to_numpy(),axis=1);df['drug_a']=pairs[:,0];df['drug_b']=pairs[:,1]
    df=df.loc[df.drug_a.ne(df.drug_b)].copy()
    df['weighted_score']=df.synergy*df.n_measurements
    df=df.groupby(['drug_a','drug_b','cell_line'],as_index=False).agg(weighted_score=('weighted_score','sum'),n_measurements=('n_measurements','sum'))
    df['synergy']=df.weighted_score/df.n_measurements
    old=json.loads(Path('results/run_config.json').read_text())['held_out_mapping']
    held={label:(name,mapping[name]) for label,name in old.items() if name in mapping and mapping[name] in drugs(df)}
    audit={'input_clean_rows':len(full),'eligible_before_structure_aggregation':int(eligible.sum()),'cohort_rows':len(df),'resolved_names':len(mapping),'structures':len(vectors),'cell_lines':df.cell_line.nunique(),'extreme_rows':int(df.synergy.abs().gt(100).sum()),'held_out':held,'unresolved_held_out':[x for x in old.values() if x not in mapping],
      'cohort_note':'Top 100 drug names by measurement count; both drugs must resolve. Known malaria labels excluded. Remaining cell identities not independently curated for disease or species. Parent-connectivity aliases grouped before splitting.'}
    (out/'cohort.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)
    df.to_csv(out/'cohort.csv.gz',index=False)
    rows=[];splits=[]
    for label in ['random',*held]:
        train,test=random_split(df,seed=42) if label=='random' else unseen_drug_split(df,[held[label][1]])
        if label!='random':assert held[label][1] not in drugs(train)
        ident=IdentityFeatures().fit(train)
        cell=OneHotEncoder(handle_unknown='ignore').fit(train[['cell_line']])
        ct=cell.transform(train[['cell_line']]);cz=cell.transform(test[['cell_line']])
        pt=pair_matrix(train,vectors);pz=pair_matrix(test,vectors)
        scale=StandardScaler(with_mean=False).fit(pt)
        reps={'identity':(ident.transform(train),ident.transform(test)),
              'chemical_cell':(hstack([scale.transform(pt),ct],format='csr'),hstack([scale.transform(pz),cz],format='csr')),
              'chemical_only':(scale.transform(pt),scale.transform(pz)),'cell_only':(ct,cz)}
        specs=[('mean','identity','mean',False),('median','identity','median',False)]
        for rep in ['identity','chemical_cell']:
            for kind in ['ridge','xgboost']:
                for clip in [False,True]:specs.append((f'{kind}_{rep}'+('_clip_train' if clip else ''),rep,kind,clip))
        specs += [('xgboost_chemical_only_clip_train','chemical_only','xgboost',True),('xgboost_cell_only_clip_train','cell_only','xgboost',True)]
        table=test[['drug_a','drug_b','cell_line','synergy']].copy()
        splits.append({'split':label,'held_out_name':held[label][0] if label!='random' else None,'n_train':len(train),'n_test':len(test),'absent_structures':sorted(drugs(test)-drugs(train))})
        for name,rep,kind,clip in specs:
            if kind in ['mean','median']:model=DummyRegressor(strategy=kind)
            elif kind=='ridge':model=Ridge(alpha=10,solver='lsqr')
            else:model=XGBRegressor(n_estimators=120,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,tree_method='hist',n_jobs=4,random_state=42,objective='reg:squarederror')
            X,Z=reps[rep];start=time.time();model.fit(X,train.synergy.clip(-100,100) if clip else train.synergy);pred=model.predict(Z)
            assert np.isfinite(pred).all(); table[name]=pred
            groups={'all':np.ones(len(test),bool),'central_abs_le_100':test.synergy.abs().le(100).to_numpy()}
            if label!='random':groups['exactly_one_unseen']=test.n_unseen_drugs.eq(1).to_numpy()
            for subset,mask in groups.items():
                if mask.any():rows.append({'split':label,'held_out':held[label][0] if label!='random' else None,'subset':subset,'model':name,**metrics(test.synergy.to_numpy()[mask],pred[mask])})
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print(label,name,round(time.time()-start,2),flush=True)
        table.to_csv(out/f'{label}_predictions.csv.gz',index=False)
    (out/'split_audit.json').write_text(json.dumps(splits,indent=2))
    r=pd.DataFrame(rows);r['evaluation']=np.where(r.split.eq('random'),'random','lodo_macro')
    summary=r.groupby(['evaluation','subset','model'])[['rmse','mae','r2','pearson','spearman']].mean()
    summary.to_csv(out/'summary.csv');print(summary.to_string(),flush=True)
if __name__=='__main__':main()
