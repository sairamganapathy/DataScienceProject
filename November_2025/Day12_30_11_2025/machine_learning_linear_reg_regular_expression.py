import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("house_price_bd.csv")
df.dropna(inplace=True)
print(df["Floor_no"].dtypes)
print(df["Floor_no"].unique())
df["Floor_no"] = df["Floor_no"].astype(str).str.extract(r'(\d+)').astype(int)
print(df["Floor_no"])
df.dropna(subset = ["Floor_no"],inplace=True)
print(df["Floor_no"])