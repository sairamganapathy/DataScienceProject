import pandas as pd
df = pd.read_csv("diamonds.csv")

unique_data = df["cut"].unique()
category_count = len(unique_data)
mapping = {}

for i in range(category_count):
    mapping[unique_data[i]] = i

print(mapping)
