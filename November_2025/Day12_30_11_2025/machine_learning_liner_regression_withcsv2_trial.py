import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("linear_house_pricing.csv")
df.dropna(inplace=True)
df["floors"] = df["floors"].replace(r"(st|nd|rd|th)$", "", regex=True)
df["price"] = df["price"].replace({"৳":"",",":""}, regex=True).astype(int)
print(df.dtypes)
df[["country","city","street"]] = OrdinalEncoder().fit_transform(df[["country","city","street"]])
x = df[["bedrooms","bathrooms","sqft_living","floors","condition", "view","yr_built","yr_renovated","city","country","street"]]
y=df["price"]
model = LinearRegression().fit(x,y)
model_prediction = model.predict(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
accuracy = r2_score(y_test, model.predict(x_test))
print(accuracy)