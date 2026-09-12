"""Cache PubChem name resolution for a predefined high-coverage drug cohort."""
import json,time,urllib.request,urllib.parse,subprocess
from pathlib import Path
import pandas as pd
from .data import load_clean

def main():
    df,_=load_clean('data/raw/drugcombs_scored.csv')
    counts=pd.concat([df.drug_a,df.drug_b]).value_counts()
    names=counts.sort_values(ascending=False,kind='stable').head(100).index.tolist()
    cache=Path('data/raw/pubchem');cache.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,name in enumerate(names):
        path=cache/f'{urllib.parse.quote(name,safe="")}.json'
        url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'+urllib.parse.quote(name,safe='')+'/property/IsomericSMILES,InChIKey/JSON'
        data=json.loads(path.read_text()) if path.exists() else {}
        if not data or 'error' in data:
            try:
                result=subprocess.run(['curl','--silent','--show-error','--fail','--location','--max-time','15',url],capture_output=True,text=True,check=True)
                data=json.loads(result.stdout)
            except Exception as e:data={'error':str(e)}
            path.write_text(json.dumps(data));time.sleep(.25)
        props=data.get('PropertyTable',{}).get('Properties',[])
        if len(props)==1:
            x=props[0];rows.append({'drug':name,'cid':x['CID'],'smiles':x.get('SMILES',x.get('IsomericSMILES',x.get('ConnectivitySMILES'))),'inchikey':x.get('InChIKey'),'source_url':url,'status':'resolved'})
        else:rows.append({'drug':name,'status':'unresolved_or_ambiguous','source_url':url})
        print(i+1,name,rows[-1]['status'],flush=True)
    pd.DataFrame(rows).to_csv('data/processed/drug_structures.csv',index=False)
if __name__=='__main__':main()
