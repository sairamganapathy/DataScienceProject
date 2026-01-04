import pandas as pd
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv("diamonds.csv")
onehotencoder = OneHotEncoder(sparse_output=False)
encoded_df = onehotencoder.fit_transform(df[['cut']])
# print(encoded_df)

new_df = pd.DataFrame(encoded_df, columns = onehotencoder.get_feature_names_out())
# print(new_df)

concat_df = pd.concat([df,new_df], axis = 1)
concat_df.drop(columns = ['cut'], inplace = True)
print(concat_df)
