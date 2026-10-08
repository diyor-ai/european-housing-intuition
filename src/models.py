"""Model definitions. Stage 2: baselines only."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

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
