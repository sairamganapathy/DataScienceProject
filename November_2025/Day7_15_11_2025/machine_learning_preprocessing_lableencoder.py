import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("diamonds.csv")
label_encoder = LabelEncoder()
df["cut"] = label_encoder.fit_transform(df["cut"])
print(df)
assigned_val = label_encoder.classes_
print(assigned_val)
mapping = {lable:index for index, lable in enumerate(assigned_val)}
print("Mapping(category --> number):", mapping)
reverse_mapping = {index:lable for index, lable in enumerate(assigned_val)}
print("Reverse(category --> number):", reverse_mapping)