import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from traitlets.utils.descriptions import describe

iris_data = load_iris()
x = iris_data['data']
y = iris_data['target']
# print(type(x))
# print(describe(x))
# print(iris_data['target'])
print(iris_data['feature_names'])
print(iris_data['target_names'])
df_x = pd.DataFrame(x, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df_y = pd.DataFrame(y)
# print(df_x.describe())
print(df_y)
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=42)
model = LogisticRegression(penalty='l2')
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print("accuracy score",accuracy_score(y_test,y_predict))
