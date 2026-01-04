import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("house_price_bd.csv")
df.dropna(inplace=True)
print(df["Floor_no"].dtypes)
print(df["Floor_no"].unique())
for data in df["Floor_no"].unique():
    if not str(data).isnumeric():
        print(data)