import pandas as pd

df = pd.read_csv("diamonds.csv")
# print(df)

encoded_df = pd.get_dummies(df['cut'])
# print(encoded_df)

final_df = pd.concat([df, encoded_df], axis=1)
# print(final_df)

final_df.drop(columns = ['cut'], inplace=True)
print(final_df)