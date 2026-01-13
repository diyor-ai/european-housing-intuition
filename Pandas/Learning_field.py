import pandas as pd

df = pd.read_csv('../house-prices-advanced-regression-techniques/train.csv')


df_New = list(df.columns)

print(df_New)