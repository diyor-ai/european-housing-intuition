from pathlib import Path

import pandas as pd

# 1. Datasetni yukla (repo ildizidagi data/train.csv)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "train.csv"
df = pd.read_csv(DATA_PATH)

# 2. Ilk ko'zdan kechirish
print("Dataset ko'rinishi:")
print(df.head())
print(f"\nShape: {df.shape}")

print("\nDataset statistikasi:")
print(df.describe())

# 3. Faqat bitta ustunni statistik ko'rish
if "SalePrice" in df.columns:
    print("\nSalePrice statistikasi:")
    print(df["SalePrice"].describe())
