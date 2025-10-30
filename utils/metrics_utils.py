import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def compute_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
