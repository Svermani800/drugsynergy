import unittest
import numpy as np
import pandas as pd
from src.chemical_benchmark import structure_features,pair_matrix
from src.splits import unseen_drug_split,drugs

class ChemistryTests(unittest.TestCase):
    def test_aliases_salts_and_order(self):
        table=pd.DataFrame({'drug':['ETHANOL','ETHANOL_ALIAS','SALT','BENZENE','WATER'],
          'smiles':['CCO','OCC','CCO.Cl','c1ccccc1','O'],'cid':[1,1,2,3,4],'status':['resolved']*5})
        names,vectors,_=structure_features(table)
        self.assertEqual(names['ETHANOL'],names['ETHANOL_ALIAS'])
        self.assertEqual(names['ETHANOL'],names['SALT'])
        a,b,c=names['ETHANOL'],names['BENZENE'],names['WATER']
        df=pd.DataFrame({'drug_a':[a,b],'drug_b':[b,c],'cell_line':['L','L'],'synergy':[1,2]})
        swapped=df.copy();swapped[['drug_a','drug_b']]=df[['drug_b','drug_a']].to_numpy()
        np.testing.assert_array_equal(pair_matrix(df,vectors).toarray(),pair_matrix(swapped,vectors).toarray())
        tr,te=unseen_drug_split(df,[a]);self.assertNotIn(a,drugs(tr))
        self.assertEqual(len(te),1)
if __name__=='__main__':unittest.main()
