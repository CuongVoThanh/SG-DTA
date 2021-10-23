import numpy as np
from math import sqrt
from scipy import stats

rmse = lambda y_true, y_pred: sqrt(((y_true - y_pred)**2).mean(axis=0))
mse = lambda y_true, y_pred: ((y_true - y_pred)**2).mean(axis=0)
pearson = lambda y_true, y_pred: np.corrcoef(y_true, y_pred)[0,1]
spearman = lambda y_true, y_pred: stats.spearmanr(y_true, y_pred)[0]

def ci(y_true, y_pred):
    """ Concordance Index """ 
    ind = np.argsort(y_true)
    y_true = y_true[ind]
    y_pred = y_pred[ind]
    g = np.subtract(np.expand_dims(y_pred, -1), y_pred)
    g = (g == 0.0).astype(np.float32) * 0.5 + (g > 0.0).astype(np.float32)
    f = np.subtract(np.expand_dims(y_true, -1), y_true) > 0.0
    f = np.tril(f.astype(np.float32), -1)
    g = np.sum(np.multiply(g, f))
    f = np.sum(f)
    return 0.0 if g == 0.0 else g/f