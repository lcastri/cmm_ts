from enum import Enum
import numpy as np


class Metric(Enum):
    NRMSEmean = {"name": "NRMSE", "value" : "NRMSEmean"}
    NRMSEminmax = {"name": "NRMSE", "value" : "NRMSEminmax"}
    NRMSEstd = {"name": "NRMSE", "value" : "NRMSEstd"}
    NRMSEiq = {"name": "NRMSE", "value" : "NRMSEiq"}
    NMAEmean = {"name": "NMAE", "value" : "NMAEmean"}
    NMAEminmax = {"name": "NMAE", "value" : "NMAEminmax"}
    NMAEstd = {"name": "NMAE", "value" : "NMAEstd"}
    NMAEiq = {"name": "NMAE", "value" : "NMAEiq"}
    RMSE = {"name": "RMSE", "value" : "rmse"}
    MSE = {"name": "MSE", "value" : "mse"}
    MAE = {"name": "MAE", "value" : "mae"}


def evaluate(mode, y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(abs(y_true - y_pred))

    if mode == Metric.NRMSEmean:
        return rmse/np.mean(y_true)

    elif mode == Metric.NRMSEminmax:
        return rmse/(np.max(y_true) - np.min(y_true))

    elif mode == Metric.NRMSEstd:
        return rmse/np.std(y_true)

    elif mode == Metric.NRMSEiq:
        return rmse/(np.quantile(y_true, 0.75) - np.quantile(y_true, 0.25))
    
    elif mode == Metric.NMAEmean:
        return mae/np.mean(y_true)

    elif mode == Metric.NMAEminmax:
        return mae/(np.max(y_true) - np.min(y_true))

    elif mode == Metric.NMAEstd:
        return mae/np.std(y_true)

    elif mode == Metric.NMAEiq:
        return mae/(np.quantile(y_true, 0.75) - np.quantile(y_true, 0.25))

    elif mode == Metric.RMSE:
        return rmse

    elif mode == Metric.MSE:
        return np.mean((y_true - y_pred)**2)
    
    elif mode == Metric.MAE:
        return mae