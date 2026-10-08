import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.data import outlier_mask
from src.evaluate import cross_validate_model, make_kfold
from src.features import GroupMedianImputer
from src.models import OutlierDropRegressor, make_preprocessing


class SpyRegressor(RegressorMixin, BaseEstimator):
    """Records the index of every row it is fitted on (class-level, survives clone())."""

    fitted_on: list = []

    def fit(self, X, y):
        SpyRegressor.fitted_on.append(set(X.index))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_preprocessing_statistics_come_from_train_only(split):
    X_train, X_test, _, _ = split
    prep = make_preprocessing().fit(X_train)
    scaler = prep.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    idx = list(prep.named_steps["columns"].transformers_[0][2]).index("GrLivArea")
    assert np.isclose(scaler.mean_[idx], X_train["GrLivArea"].mean())
    full_mean = pd.concat([X_train, X_test])["GrLivArea"].mean()
    assert not np.isclose(scaler.mean_[idx], full_mean)


def test_fitting_ignores_test_rows(split):
    X_train, X_test, _, _ = split
    prep_a = make_preprocessing().fit(X_train)
    prep_b = make_preprocessing().fit(X_train)
    # transforming (very different) test rows must not change what was learned in fit
    prep_b.transform(X_test.assign(GrLivArea=10**6))
    sa = prep_a.named_steps["lotfrontage"].group_medians_
    sb = prep_b.named_steps["lotfrontage"].group_medians_
    assert sa == sb
    assert GroupMedianImputer().fit(X_train).global_median_ == X_train["LotFrontage"].median()


def test_cv_never_fits_on_validation_rows(split):
    X_train, _, y_train, _ = split
    SpyRegressor.fitted_on = []
    cross_validate_model(SpyRegressor(), X_train, y_train, n_splits=5)
    folds = list(make_kfold(5).split(X_train))
    assert len(SpyRegressor.fitted_on) == 5
    for fitted, (tr, va) in zip(SpyRegressor.fitted_on, folds):
        assert fitted == set(X_train.index[tr])
        assert fitted.isdisjoint(X_train.index[va])


def test_outliers_dropped_from_fit_only(split):
    X_train, X_test, y_train, _ = split
    bad = set(X_train.index[outlier_mask(X_train, y_train)])
    assert len(bad) == 2
    SpyRegressor.fitted_on = []
    model = OutlierDropRegressor(SpyRegressor(), drop_outliers=True).fit(X_train, y_train)
    assert SpyRegressor.fitted_on[0].isdisjoint(bad)
    assert len(SpyRegressor.fitted_on[0]) == len(X_train) - 2
    assert len(model.predict(X_train)) == len(X_train)  # predict never filters rows
    SpyRegressor.fitted_on = []
    OutlierDropRegressor(SpyRegressor(), drop_outliers=False).fit(X_train, y_train)
    assert len(SpyRegressor.fitted_on[0]) == len(X_train)
