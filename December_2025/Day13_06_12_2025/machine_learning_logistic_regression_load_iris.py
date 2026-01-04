import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
loaded_iris = load_iris()
x = loaded_iris.data
y = loaded_iris.target
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
model = LogisticRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
# print("classification report",classification_report(y_test,y_predict))
print("\n accuracy score",accuracy_score(y_test,y_predict))
iris_data = [[15,3.1,4.1,3.2]]
prediction = model.predict(iris_data)
print(prediction)
#
# confusion_matrix = confusion_matrix(y_test,y_predict)
# print(confusion_matrix)