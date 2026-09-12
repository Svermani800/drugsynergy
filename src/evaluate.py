import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    variable=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
    return {"n_test":len(y),"rmse":float(np.sqrt(mean_squared_error(y,p))),
            "mae":float(mean_absolute_error(y,p)),
            "r2":float(r2_score(y,p)) if len(y)>1 and np.ptp(y)>0 else None,
            "pearson":float(pearsonr(y,p).statistic) if variable else None,
            "spearman":float(spearmanr(y,p).statistic) if variable else None}
