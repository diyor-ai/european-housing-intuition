"""Metrics and cross-validation helper (train data only)."""
import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.data import RANDOM_STATE

N_SPLITS = 5


def rmsle(y_true, y_pred) -> float:
    """Kaggle's metric: RMSE on log1p prices. Negative predictions are clipped to 0."""
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def all_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "RMSLE": rmsle(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def make_kfold(n_splits: int = N_SPLITS) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def cross_validate_model(model, X, y, n_splits: int = N_SPLITS) -> dict:
    """K-fold CV; the whole pipeline is re-fitted on each fold's training part.

    Returns mean and std of every metric over folds, plus mean fit time (s).
    """
    kf = make_kfold(n_splits)
    rows, times, n_neg = [], [], 0
    for tr, va in kf.split(X):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X.iloc[tr], y.iloc[tr])
        times.append(time.perf_counter() - t0)
        pred = m.predict(X.iloc[va])
        n_neg += int((pred < 0).sum())
        rows.append(all_metrics(y.iloc[va], pred))
    out = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows])
        out[k] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    out["R2_worst_fold"] = float(min(r["R2"] for r in rows))
    out["n_negative_preds"] = n_neg
    out["fit_time"] = float(np.mean(times))
    return out
