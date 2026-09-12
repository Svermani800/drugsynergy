"""Sum shared drug one-hot vectors: predictions are invariant to drug order."""
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class IdentityFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.drug_encoder_ = OneHotEncoder(handle_unknown="ignore")
        names = pd.concat([X.drug_a,X.drug_b],ignore_index=True).to_numpy().reshape(-1,1)
        self.drug_encoder_.fit(names)
        self.cell_encoder_ = OneHotEncoder(handle_unknown="ignore").fit(X[["cell_line"]])
        return self
    def transform(self, X):
        a=self.drug_encoder_.transform(X.drug_a.to_numpy().reshape(-1,1))
        b=self.drug_encoder_.transform(X.drug_b.to_numpy().reshape(-1,1))
        c=self.cell_encoder_.transform(X[["cell_line"]])
        return hstack([a+b,c],format="csr")
