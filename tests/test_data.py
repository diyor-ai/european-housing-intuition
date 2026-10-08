from src.data import TARGET, load_data, outlier_mask, split_data


def test_shape_and_key_columns(df):
    assert df.shape == (1460, 81)
    for col in [TARGET, "GrLivArea", "OverallQual", "Neighborhood", "LotFrontage", "YrSold"]:
        assert col in df.columns
    assert df[TARGET].notna().all()


def test_none_category_preserved(df):
    # "None" is a real MasVnrType category; pandas' default NA list would turn it into NaN
    assert (df["MasVnrType"] == "None").sum() > 800
    assert df["MasVnrType"].isna().sum() < 20


def test_nan_means_absent_columns_are_nan(df):
    assert df["PoolQC"].isna().sum() == 1453
    assert df["LotFrontage"].isna().sum() == 259


def test_split_is_deterministic_and_disjoint(df):
    a = split_data(df)
    b = split_data(df)
    assert a[0].index.equals(b[0].index) and a[1].index.equals(b[1].index)
    assert len(a[0]) == 1168 and len(a[1]) == 292
    assert set(a[0].index).isdisjoint(a[1].index)


def test_split_has_no_target_or_id_in_features(split):
    X_train, X_test, _, _ = split
    for X in (X_train, X_test):
        assert TARGET not in X.columns and "Id" not in X.columns


def test_outlier_mask_finds_the_two_known_houses(df):
    X = df.drop(columns=[TARGET])
    assert outlier_mask(X, df[TARGET]).sum() == 2
    assert load_data().shape == df.shape  # cached reload is identical in shape
