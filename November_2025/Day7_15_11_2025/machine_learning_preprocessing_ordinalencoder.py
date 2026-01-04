import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
df = pd.read_csv("diamonds.csv")

ordinal_encoder = OrdinalEncoder()
# df["cut"] = ordinal_encoder.fit_transform(df[["cut"]])
df["cut"] =ordinal_encoder.fit_transform(df[["cut"]])
print(df["cut"])