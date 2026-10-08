"""Feature engineering. Every function is row-wise (no statistics learned from data),
so it cannot leak information between train and test. The only learned step,
GroupMedianImputer, is fitted inside the sklearn Pipeline on training rows only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Quality scale Po < Fa < TA < Gd < Ex; "None" (feature absent) ranks below Po.
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
# Categorical columns where NaN means "the house has no such feature" (see EDA).
# MasVnrType and Electrical are NOT here: their NaNs are truly missing.
ABSENT_COLS = [
    "Alley", "Fence", "MiscFeature", "GarageType", "GarageFinish",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
] + QUALITY_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered features added. Safe to call on any split."""
    out = df.copy()

    # NaN = feature absent -> explicit category (keeps the information instead of imputing it away)
    for col in ABSENT_COLS:
        if col in out:
            out[col] = out[col].fillna("None")

    # Total living space: buyers pay for all floors + basement, not one of them
    out["TotalSF"] = (
        out["TotalBsmtSF"].fillna(0) + out["1stFlrSF"] + out["2ndFlrSF"]
    )
    # Age at sale: price depends on how old the house was when sold, not on the build year itself
    out["HouseAge"] = (out["YrSold"] - out["YearBuilt"]).clip(lower=0)
    # Recent renovation raises value; 0 means newly built or just remodelled
    out["YearsSinceRemodel"] = (out["YrSold"] - out["YearRemodAdd"]).clip(lower=0)
    # Bathrooms are split across 4 columns; half baths count as 0.5
    out["TotalBath"] = (
        out["FullBath"] + 0.5 * out["HalfBath"]
        + out["BsmtFullBath"].fillna(0) + 0.5 * out["BsmtHalfBath"].fillna(0)
    )
    # Existence flags: a feature's presence can matter more than its size (and area=0 is a spike)
    out["HasGarage"] = (out["GarageArea"].fillna(0) > 0).astype(int)
    out["HasPool"] = (out["PoolArea"] > 0).astype(int)
    out["HasFireplace"] = (out["Fireplaces"] > 0).astype(int)

    # Quality grades are ordered: encode as 0..5 so models can use the order
    for col in QUALITY_COLS:
        out[col] = out[col].map(QUALITY_MAP).fillna(0).astype(int)

    # MSSubClass is a building-type code (20, 60, ...), not a quantity -> treat as category
    out["MSSubClass"] = out["MSSubClass"].astype(str)
    return out


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill NaN in `column` with the median of its `group` (learned on training rows only).

    Used for LotFrontage by Neighborhood: lot width is similar within a neighborhood.
    Unseen/NaN groups fall back to the global training median.
    """

    def __init__(self, column="LotFrontage", group="Neighborhood"):
        self.column = column
        self.group = group

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(self.group)[self.column].median().to_dict()
        self.global_median_ = X[self.column].median()
        return self

    def transform(self, X):
        out = X.copy()
        fill = out[self.group].map(self.group_medians_).fillna(self.global_median_)
        out[self.column] = out[self.column].fillna(fill)
        return out
