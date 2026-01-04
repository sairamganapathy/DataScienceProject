import pandas as pd
df = pd.read_csv("diamonds.csv")
# df['cut']= df['cut'].fillna(100)
# print(df)
# df[['carat','price']]= df[['carat','price']].fillna(10)
# print(df)
#below is just a trial
# df.fillna(20)#not working
# print(df)
# count_of_null = df.isnull().sum()
# print(count_of_null)
# df['cut'] = df['cut'].ffill()
# print(df)
# 
# df['color'] = df['color'].bfill()
# print(df)

print(df)
df['price'] = df['price'].fillna(df['x'].mean())
print(df)