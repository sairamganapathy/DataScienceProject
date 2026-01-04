import pandas as pd
df = pd.read_csv("diamonds.csv")
data_types = df.dtypes
columns = df.columns
num_columns = []
categorical_columns = []

for column in columns:
    if data_types[column] == "object":
        num_columns.append(column)
    else:
        categorical_columns.append(column)

print(num_columns)
print(categorical_columns)