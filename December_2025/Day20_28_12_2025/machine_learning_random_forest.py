from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3)
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)
selector = SelectFromModel(rf, prefit = True)
X_train_new = selector.transform(X_train)
X_test_new = selector.transform(X_test)
rf_new = RandomForestClassifier(n_estimators =100, random_state = 42)
rf_new.fit(X_train_new, y_train)
y_pred = rf_new.predict(X_test_new)
acc = accuracy_score(y_test, y_pred)
print(f"test accuracy: {acc:.3f}")
print(f"test accuracy: {acc}")