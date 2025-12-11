import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# -------------------------
# Extra Regression Metrics
# -------------------------
def regression_metrics(y_true, y_pred):
    """
    Compute regression metrics including RMSE%, PBIAS%, and ubRMSE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mean_obs = np.mean(y_true)
    perc_rmse = (rmse / mean_obs) * 100 if mean_obs != 0 else np.nan
    pbias = 100 * np.sum(y_pred - y_true) / np.sum(y_true) if np.sum(y_true) != 0 else np.nan
    ubrmse = np.sqrt(np.mean(((y_pred - y_true) - np.mean(y_pred - y_true))**2))

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "RMSE%": perc_rmse,
        "PBIAS%": pbias,
        "ubRMSE": ubrmse,
    }


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
# Define function to compute metrics
def compute_metrics(group):
    y_true = group['MODIS_Albedo_WSA_shortwave']
    y_pred = group['RF_Predicted']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_percent = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    return pd.Series({'RMSE_percent': rmse_percent, 'R2': r2})