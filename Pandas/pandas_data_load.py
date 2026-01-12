import pandas as pd

# CSV yuklash
df = pd.read_csv("../house-prices-advanced-regression-techniques/train.csv")  # yoki Eurostat CSV
print("DataFrame yuklandi. Shape:", df.shape)  # (row soni, column soni)
print("Birinchi 5 qator:")
print(df.head())  # 5 ta birinchi qator
print("\nOxirgi 5 qator:")
print(df.tail())  # Oxirgi 5 qator

print("Asosiy statistika:")
print(df.describe())  # mean, std, min, max, quartiles

# Missing values check
print("Missing values soni:", df.isnull().sum())

# Column types
print("Column turlari:\n", df.dtypes)

# Narx column statistikasi (agar 'SalePrice' bo'lsa)
if 'SalePrice' in df.columns:
    print("Narx statistikasi:\n", df['SalePrice'].describe())