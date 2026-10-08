"""Load the Ames Housing dataset (Kaggle "House Prices: Advanced Regression Techniques").

Primary source: OpenML via sklearn (cached under data/).
Fallback: data/train.csv downloaded manually from Kaggle.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_FILE = DATA_DIR / "ames_openml.csv"
KAGGLE_FILE = DATA_DIR / "train.csv"
# "None" is a real category (e.g. MasVnrType); only "NA"/"" mean missing.
# pandas' default NA list would silently turn "None" into NaN.
CSV_KWARGS = {"keep_default_na": False, "na_values": ["NA", ""]}
TARGET = "SalePrice"


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


if __name__ == "__main__":
    frame = load_data()
    print(f"Shape: {frame.shape}")
    print(f"SalePrice median: ${frame[TARGET].median():,.0f}")
