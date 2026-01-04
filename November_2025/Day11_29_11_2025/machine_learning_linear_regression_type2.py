import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

#testing with existing data instead of testing with new data
data = { "year": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
         "price":[20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000]}
df = pd.DataFrame(data)
# print(df)
x = df[["year"]]#always x should be in 2 dim
y = df["price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
# print(x_train)
# print(x_test)
model = LinearRegression()#model declaration
model.fit(x_train,y_train)#training

model_prediction = model.predict(x_test)

accuracy = r2_score(y_test, model_prediction)
print(accuracy)
# print(model_prediction)
#
# # input_data['price']= model_prediction
# # print(input_data)
# # input_data['price'] = np.array(input_data['price'])
# # print(input_data)
# predicted_data = {"price":model_prediction}
# final_df = pd.DataFrame(predicted_data)
# total_df = pd.concat([input_df,final_df], axis = 1)
# print(total_df)
# complete_df = pd.concat([df,total_df], axis = 0)
# print(complete_df)