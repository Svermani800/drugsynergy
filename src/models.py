from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from .features import IdentityFeatures

def baseline_models():
    # Fixed alpha: no tuning on the outer test set. Ridge is regularized linear regression.
    return {"mean":make_pipeline(IdentityFeatures(),DummyRegressor(strategy="mean")),
            "ridge_identity":make_pipeline(IdentityFeatures(),Ridge(alpha=10.0,solver="lsqr"))}
