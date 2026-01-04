import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("house_price_bd.csv")
df.dropna(inplace=True)
# df["Floor_no"] = df["Floor_no"].replace(r"(st|nd|rd|th)$", "", regex=True)#if we know the text
df["Floor_no"] =df["Floor_no"].astype(str).str.extract(r'(\d+)').astype(int)#if we know there can be any string
df.dropna(subset = ["Floor_no"],inplace=True)

df["Price_in_taka"] = df["Price_in_taka"].replace({"৳":"",",":""}, regex=True).astype(int)
print(df.dtypes)
df[["Bedrooms","Floor_no","Floor_area","City","Price_in_taka"]].drop_duplicates()
x = df[["Bedrooms","Bathrooms", "Floor_area"]]#
y=df["Price_in_taka"]
model = LinearRegression().fit(x,y)
model_prediction = model.predict(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
df_concat = pd.concat([x_test,y_test],axis=1)
# print(df_concat)
accuracy = r2_score(y_test, model.predict(x_test))
print(accuracy)

mean_square_err = mean_squared_error(y_test, model.predict(x_test))
print(mean_square_err)