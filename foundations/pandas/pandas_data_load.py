import pandas as pd

# 1. Datasetni yukla
df = pd.read_csv('../house-prices-advanced-regression-techniques/train.csv')

# 2. Ilk ko'zdan kechirish
print("Dataset ko'rinishi:")
print(df.head())

print("\nDataset statistikasi:")
print(df.describe())

# 3. Faqat bitta ustunni statistik ko'rish
if 'price' in df.columns:
    print("\nNarxlar statistikasi:")
    print(df['price'].describe())