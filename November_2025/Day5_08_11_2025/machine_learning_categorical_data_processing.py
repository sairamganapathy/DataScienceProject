import pandas as pd
df = pd.read_csv("diamonds.csv")
cut_category = df['cut'].value_counts()
print(cut_category)
unique_cut_category = df['cut'].unique()
print(unique_cut_category)
total_unique_cut_category = df['cut'].nunique()
print(total_unique_cut_category)