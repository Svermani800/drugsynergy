import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_clean
from src.splits import unseen_drug_split, random_split, drugs
from src.models import baseline_models
from src.evaluate import metrics

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame({'drug_a':['A','B','C','A','B','C'],
          'drug_b':['B','X','X','C','C','D'],'cell_line':['L']*6,'synergy':[1.,2.,3.,4.,5.,6.]})
    def test_both_columns_excluded(self):
        train,test=unseen_drug_split(self.df,['c'])
        self.assertNotIn('C',drugs(train)); self.assertEqual(len(test),4)
        self.assertEqual(len(train)+len(test),len(self.df))
    def test_group_holdout(self):
        train,test=unseen_drug_split(self.df,['X','D'])
        self.assertFalse(drugs(train)&{'X','D'})
        self.assertTrue(test.n_unseen_drugs.ge(1).all())
    def test_two_unseen(self):
        extra=pd.DataFrame({'drug_a':['X'],'drug_b':['D'],'cell_line':['L'],'synergy':[1.]})
        _,test=unseen_drug_split(pd.concat([self.df,extra],ignore_index=True),['X','D'])
        self.assertEqual(test.iloc[-1].n_unseen_drugs,2)
    def test_unknown_or_empty_split(self):
        for held in [[],['Z'],['A','B','C','D','X']]:
            with self.assertRaises(ValueError): unseen_drug_split(self.df,held)
    def test_fit_only_training_and_order_invariance(self):
        train,test=unseen_drug_split(self.df,['X'])
        model=baseline_models()['ridge_identity'].fit(train,train.synergy)
        self.assertNotIn('X',model.steps[0][1].drug_encoder_.categories_[0])
        swapped=test.copy(); swapped[['drug_a','drug_b']]=test[['drug_b','drug_a']].to_numpy()
        np.testing.assert_allclose(model.predict(test),model.predict(swapped))
    def test_clean_and_repeats(self):
        raw=pd.DataFrame({'Drug1':[' a ','B','A','A','A','A'],
            'Drug2':['b','A','A','C','D',None], 'Cell line':['l']*6,
            'Bliss':[2,4,5,'oops',np.inf,9]})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.csv'; raw.to_csv(path,index=False)
            clean,report=load_clean(path)
        self.assertEqual(len(clean),1); self.assertEqual(clean.synergy.iloc[0],3)
        self.assertEqual(report['invalid_rows'],3)
        self.assertEqual(clean.n_measurements.iloc[0],2)
    def test_random_disjoint_and_reproducible(self):
        tr,te=random_split(self.df); tr2,te2=random_split(self.df)
        self.assertFalse(set(tr.index)&set(te.index)); self.assertEqual(list(te.index),list(te2.index))
        with self.assertRaises(ValueError): random_split(pd.concat([self.df,self.df]))
    def test_constant_correlation_is_undefined(self):
        self.assertIsNone(metrics([1,2,3],[2,2,2])['pearson'])
if __name__=='__main__': unittest.main()
