"""Load the Ames Housing dataset (Kaggle "House Prices: Advanced Regression Techniques").

Primary source: OpenML via sklearn (cached under data/).
Fallback: data/train.csv downloaded manually from Kaggle.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_FILE = DATA_DIR / "ames_openml.csv"
KAGGLE_FILE = DATA_DIR / "train.csv"
# "None" is a real category (e.g. MasVnrType); only "NA"/"" mean missing.
# pandas' default NA list would silently turn "None" into NaN.
CSV_KWARGS = {"keep_default_na": False, "na_values": ["NA", ""]}
TARGET = "SalePrice"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _from_openml() -> pd.DataFrame:
    bunch = fetch_openml(name="house_prices", as_frame=True, data_home=DATA_DIR / "openml")
    return bunch.frame


def load_data() -> pd.DataFrame:
    """Return the full labelled dataset (1,460 rows incl. SalePrice)."""
    DATA_DIR.mkdir(exist_ok=True)
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE, **CSV_KWARGS)
    try:
        df = _from_openml()
        df.to_csv(CACHE_FILE, index=False)
        return df
    except Exception as exc:  # network down, OpenML change, ...
        if KAGGLE_FILE.exists():
            print(f"OpenML failed ({exc!r}); falling back to {KAGGLE_FILE}")
            return pd.read_csv(KAGGLE_FILE, **CSV_KWARGS)
        raise FileNotFoundError(
            "Could not fetch from OpenML and data/train.csv is missing. "
            "Download train.csv from Kaggle into data/."
        ) from exc


def outlier_mask(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """The 2 known 'huge but cheap' houses (GrLivArea > 4000 sqft, price < $300k).

    Both are partial sales (see notebooks/01_eda.ipynb).
    """
    return (X["GrLivArea"] > 4000) & (y < 300_000)


def split_data(df: pd.DataFrame):
    """The one fixed train/test split (80/20, random_state=42).

    Returns X_train, X_test, y_train, y_test. The test set must only be
    touched once, at the very end.
    """
    X = df.drop(columns=[TARGET, "Id"], errors="ignore")
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


if __name__ == "__main__":
    frame = load_data()
    print(f"Shape: {frame.shape}")
    X_tr, X_te, _, _ = split_data(frame)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"SalePrice median: ${frame[TARGET].median():,.0f}")
