import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data, split_data  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return load_data()


@pytest.fixture(scope="session")
def split(df):
    return split_data(df)
