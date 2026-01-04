import pandas as pd
df = pd.read_csv("diamonds.csv")
# unique_cut = df["cut"].unique()
# print(unique_cut)
mapping = {
    'Ideal':1,
    'Premium':2,
    'Good':4,
    'Very Good':5,
    'Fair':6}
df["cut"] = df["cut"].fillna(0)
# df["cut_no"] = df["cut"].map(mapping)
# df["cut"] = df["cut"].fillna(0)
# print(df)
df["cut"] = df["cut"].map(mapping)
# df["cut"] = df["cut"].fillna(0)
# print(df)
# df = df.replace(mapping)
# df["cut"] = df["cut"].astype('int')
print(df)