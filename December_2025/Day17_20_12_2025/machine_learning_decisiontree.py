import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

my_input = load_breast_cancer()
X = my_input.data
Y = my_input.target
dataset = pd.DataFrame(X, columns = my_input.feature_names)
print(dataset.info())
dataset
sns.scatterplot(data = dataset, x = 'mean radius', y = 'mean fractal dimension', hue = Y)
plt.show()

X = dataset.copy()
y = my_input.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
model = DecisionTreeClassifier(ccp_alpha=0.01)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metric_evaluation = accuracy_score(y_test,y_pred)
print(metric_evaluation)
# print(model.get_params())
