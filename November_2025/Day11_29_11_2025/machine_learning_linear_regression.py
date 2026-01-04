import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

data = { "year": [2000, 2001, 2002, 2003, 2004],
         "price":[20000, 30000, 40000, 50000, 60000]}
df = pd.DataFrame(data)
# print(df)
x = df[["year"]]#always x should be in 2 dim
y = df["price"]

model = LinearRegression()#model declaration
model.fit(x,y)#training

input_data = {"year":[2005, 2007]}
input_df = pd.DataFrame(input_data)

model_prediction = model.predict(input_df)
# print(model_prediction)

# input_data['price']= model_prediction
# print(input_data)
# input_data['price'] = np.array(input_data['price'])
# print(input_data)
predicted_data = {"price":model_prediction}
final_df = pd.DataFrame(predicted_data)
total_df = pd.concat([input_df,final_df], axis = 1)
print(total_df)
complete_df = pd.concat([df,total_df], axis = 0)
print(complete_df)