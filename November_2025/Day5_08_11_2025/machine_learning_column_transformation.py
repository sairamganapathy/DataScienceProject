import pandas as pd
df = pd.read_csv("diamonds.csv")
columns = df.columns
print("before removal of column")
print(columns)
#df.drop(columns="Unnamed: 0", inplace = True)
df.drop(columns=["Unnamed: 0","cut"], inplace = True)
after_removal = df.columns
print("after removal of column")
print(after_removal)