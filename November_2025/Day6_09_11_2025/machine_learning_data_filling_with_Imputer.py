import pandas as pd
from sklearn.impute import SimpleImputer
df = pd.read_csv("diamonds.csv")
print(df)
# simple_impute_object = SimpleImputer()
# df['price'] = simple_impute_object.fit_transform(df[['price']])
# print(df)
# print(df['price'])
##most_frequent helps in updating the most frequent value
simple_impute_object = SimpleImputer(strategy='most_frequent')
df['price'] = simple_impute_object.fit_transform(df[['price']])
print(df)
print(df['price'])