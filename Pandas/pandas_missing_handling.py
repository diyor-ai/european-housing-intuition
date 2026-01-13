import pandas as pd

# first we load dataset
df = pd.read_csv('../house-prices-advanced-regression-techniques/train.csv')

new_df = df.dropna()

print(new_df.to_string())

# Note: By default, the dropna() method returns a new DataFrame, and will not change the original.

"""
Remove all rows with NULL values:

import pandas as pd

df = pd.read_csv('data.csv')

df.dropna(inplace = True)

print(df.to_string())

Note: Now, the dropna(inplace = True) will NOT return a new DataFrame,
but it will remove all rows containing NULL values from the original DataFrame.
"""

# Replace Empty Values
# The fillna() method allows us to replace empty cells with a value:

# df.fillna(130, inplace=True)
# print(df.to_string())

# The example above replaces all empty cells in the whole Data Frame.
# To only replace empty values for one column, specify the column name for the DataFrame:

#df.fillna({"Calories": 130}, inplace=True)

#Mean = the average value (the sum of all values divided by number of values).

x = df["SalePrice"].mean()
print(f" Mean value of SalePrice column is {x}")

#Median = the value in the middle, after you have sorted all values ascending.

y = df["SalePrice"].median()
print(f" Median value of SalePrice column is {y}")

#Mode = the value that appears most frequently.

z = df['SalePrice'].mode()[0]
df.fillna({'SalePrice': z}, inplace=True)
print(f" Mode value of SalePrice column is {z}")