"""Model definitions: Stage 2 baselines and Stage 3 full pipelines.

Structure of every Stage 3 model (outer to inner):
    GridSearchCV                      <- hyper-parameters tuned by inner 5-fold CV (RMSLE)
      OutlierDropRegressor            <- optionally drops 2 outliers from the rows it is FITTED on
        TransformedTargetRegressor    <- trains on log1p(price), predicts expm1(...)
          Pipeline: add_features -> LotFrontage by Neighborhood -> ColumnTransformer -> estimator
Everything learned from data lives inside, so it is fitted on training rows only.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import RANDOM_STATE, outlier_mask
from src.evaluate import make_kfold, rmsle
from src.features import GroupMedianImputer, add_features

# 5 strong numeric features picked from the EDA correlation table
BASELINE_FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]


def baseline_median():
    """Baseline A: always predict the training-set median price."""
    return DummyRegressor(strategy="median")


def baseline_linear(log_target: bool = False):
    """Baseline B: linear regression on a few strong numeric features."""
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), BASELINE_FEATURES)], remainder="drop"
    )
    pipe = Pipeline([("pre", pre), ("lr", LinearRegression())])
    if log_target:
        return TransformedTargetRegressor(pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


class OutlierDropRegressor(RegressorMixin, BaseEstimator):
    """Drops the known outliers from the data used in fit(); predict() never drops anything.

    Placed INSIDE GridSearchCV / cross-validation, so it only ever filters the training
    part of a fold. Validation and test rows are untouched.
    """

    def __init__(self, estimator, drop_outliers=True):
        self.estimator = estimator
        self.drop_outliers = drop_outliers

    def fit(self, X, y):
        if self.drop_outliers:
            keep = ~outlier_mask(X, y).to_numpy()
            X, y = X[keep], y[keep]
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def make_preprocessing() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),  # only truly-missing NaNs remain here
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_exclude=["object", "str"])),
        ("cat", categorical, make_column_selector(dtype_include=["object", "str"])),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("lotfrontage", GroupMedianImputer()),
        ("columns", columns),
    ])


def _log_pipeline(estimator) -> TransformedTargetRegressor:
    pipe = Pipeline([("prep", make_preprocessing()), ("model", estimator)])
    return TransformedTargetRegressor(pipe, func=np.log1p, inverse_func=np.expm1)


# Path from GridSearchCV to the final estimator's parameters
_P = "estimator__regressor__model__"


def _specs(fast: bool):
    n_trees = 20 if fast else 300
    return {
        "Ridge": (Ridge(), {_P + "alpha": [3, 30] if fast else [1, 3, 10, 30, 100, 300]}),
        "Lasso": (
            Lasso(max_iter=20000),
            {_P + "alpha": [1e-3, 1e-2] if fast else [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]},
        ),
        "RandomForest": (
            RandomForestRegressor(n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=1),
            {_P + "max_features": [0.3] if fast else [0.3, 0.6],
             _P + "min_samples_leaf": [1, 3]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(max_iter=50 if fast else 300, random_state=RANDOM_STATE),
            {_P + "learning_rate": [0.1] if fast else [0.03, 0.1],
             _P + "max_depth": [3] if fast else [3, 6]},
        ),
    }


MODEL_NAMES = list(_specs(False))


def make_model(name: str, drop_outliers: bool = False, fast: bool = False, n_splits: int = 5):
    """Tuned model: GridSearchCV over the full pipeline, scored by RMSLE with inner CV."""
    estimator, grid = _specs(fast)[name]
    wrapped = OutlierDropRegressor(_log_pipeline(estimator), drop_outliers=drop_outliers)
    return GridSearchCV(
        wrapped,
        grid,
        scoring=make_scorer(rmsle, greater_is_better=False),
        cv=make_kfold(n_splits),
        n_jobs=-1,
        refit=True,
    )
