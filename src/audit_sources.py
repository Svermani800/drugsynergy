"""Compare candidate source IDs AND names, never assume IDs alone match."""
import pandas as pd,json
from pathlib import Path
p=Path('.');raw=pd.read_csv(p/'data/raw/drugcombs_scored.csv');ref=pd.read_csv(p/'data/raw/drugcomb_v1.4.csv',low_memory=False)
# IDs are candidates for a join, not proof of provenance: check identities and scores too.
m=raw.merge(ref[['block_id','drug_row','drug_col','cell_line_name','synergy_bliss']],left_on='ID',right_on='block_id',how='left',validate='one_to_one')
match=(m.Drug1.str.upper()==m.drug_row.str.upper())&(m.Drug2.str.upper()==m.drug_col.str.upper())&(m['Cell line'].str.upper()==m.cell_line_name.str.upper())
score=(m.Bliss-m.synergy_bliss).abs().le(.011)
extreme=m.Bliss.abs().gt(100)
out=p/'results/model_comparison';out.mkdir(exist_ok=True)
report={'raw_rows':len(raw),'raw_abs_bliss_gt_100':int(extreme.sum()),'reference_rows':len(ref),
'matched_id_and_names':int(match.sum()),'matched_id_names_and_bliss_within_0_011':int((match&score).sum()),
'extreme_rows_matched_names':int((extreme&match).sum()),'extreme_rows_matched_names_and_bliss':int((extreme&match&score).sum()),
'extreme_cell_counts':raw.loc[extreme,'Cell line'].value_counts().head(15).to_dict(),
'extreme_other_scores_summary':raw.loc[extreme,['ZIP','Bliss','Loewe','HSA']].describe().to_dict()}
(out/'extreme_audit.json').write_text(json.dumps(report,indent=2))
m.loc[extreme].to_csv(out/'extreme_source_matches.csv',index=False)
print(json.dumps(report,indent=2))
