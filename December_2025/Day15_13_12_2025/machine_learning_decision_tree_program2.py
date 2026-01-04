from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris_data = load_iris()
X = iris_data.data
Y = iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# metric_evaluation = accuracy_score(y_test,y_pred)
# print(metric_evaluation)
plt.figure(figsize=(20,10))
plot_tree(model, filled = True, feature_names = iris_data.feature_names, class_names = iris_data.target_names)
plt.show()