import numpy as np
import pandas as pd
import pytest

from src.data import TARGET
from src.features import ABSENT_COLS, QUALITY_MAP, GroupMedianImputer, add_features


@pytest.fixture
def rows(df):
    return df.drop(columns=[TARGET, "Id"]).head(20).copy()


def test_add_features_does_not_modify_input(rows):
    before = rows.copy()
    add_features(rows)
    pd.testing.assert_frame_equal(rows, before)


def test_total_sf(rows):
    rows.loc[rows.index[0], ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]] = [800, 1000, 500]
    rows.loc[rows.index[1], ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]] = [np.nan, 900, 0]
    out = add_features(rows)
    assert out["TotalSF"].iloc[0] == 2300
    assert out["TotalSF"].iloc[1] == 900  # missing basement counts as 0


def test_house_age_and_remodel_are_clipped_at_zero(rows):
    rows.loc[rows.index[0], ["YrSold", "YearBuilt", "YearRemodAdd"]] = [2008, 1998, 2003]
    rows.loc[rows.index[1], ["YrSold", "YearBuilt", "YearRemodAdd"]] = [2007, 2008, 2008]  # sold before built
    out = add_features(rows)
    assert out["HouseAge"].iloc[0] == 10 and out["YearsSinceRemodel"].iloc[0] == 5
    assert out["HouseAge"].iloc[1] == 0 and out["YearsSinceRemodel"].iloc[1] == 0


def test_total_bath_counts_half_baths_as_half(rows):
    rows.loc[rows.index[0], ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]] = [2, 1, 1, 1]
    out = add_features(rows)
    assert out["TotalBath"].iloc[0] == 2 + 0.5 + 1 + 0.5


def test_existence_flags(rows):
    rows.loc[rows.index[0], ["GarageArea", "PoolArea", "Fireplaces"]] = [0, 0, 0]
    rows.loc[rows.index[1], ["GarageArea", "PoolArea", "Fireplaces"]] = [400, 500, 2]
    out = add_features(rows)
    assert list(out.loc[rows.index[:2], "HasGarage"]) == [0, 1]
    assert list(out.loc[rows.index[:2], "HasPool"]) == [0, 1]
    assert list(out.loc[rows.index[:2], "HasFireplace"]) == [0, 1]


def test_quality_columns_become_ordinal(rows):
    rows["KitchenQual"] = ["Po", "Fa", "TA", "Gd", "Ex"] * 4
    rows.loc[rows.index[0], "PoolQC"] = np.nan  # NaN = no pool
    out = add_features(rows)
    assert list(out["KitchenQual"].iloc[:5]) == [1, 2, 3, 4, 5]
    assert out["PoolQC"].iloc[0] == QUALITY_MAP["None"] == 0
    assert QUALITY_MAP["Po"] < QUALITY_MAP["Fa"] < QUALITY_MAP["TA"] < QUALITY_MAP["Gd"] < QUALITY_MAP["Ex"]


def test_absent_nan_becomes_none_category_but_missing_stays_nan(rows):
    rows.loc[rows.index[0], ["Alley", "MasVnrType"]] = np.nan
    out = add_features(rows)
    assert out["Alley"].iloc[0] == "None"
    assert pd.isna(out["MasVnrType"].iloc[0])  # truly missing -> imputed later inside the pipeline
    assert not out[[c for c in ABSENT_COLS if c in out and out[c].dtype == object]].isna().any().any()


def test_mssubclass_becomes_categorical(rows):
    assert add_features(rows)["MSSubClass"].map(type).eq(str).all()


def test_group_median_imputer_uses_group_then_global_median():
    train = pd.DataFrame({"Neighborhood": ["A", "A", "A", "B", "B"],
                          "LotFrontage": [10.0, 20.0, np.nan, 100.0, 300.0]})
    imp = GroupMedianImputer().fit(train)
    test = pd.DataFrame({"Neighborhood": ["A", "B", "Z"], "LotFrontage": [np.nan] * 3})
    out = imp.transform(test)
    assert list(out["LotFrontage"]) == [15.0, 200.0, 60.0]  # global train median of [10,20,100,300] = 60
