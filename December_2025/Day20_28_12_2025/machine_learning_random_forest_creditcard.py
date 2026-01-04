import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
df = pd.read_csv("creditcard_for_random_Forest.csv")
target_finding = df['Class'].unique()
column_check = df.columns
X = df.drop("Class",axis = 1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42, test_size = 0.3)
model = RandomForestClassifier(n_estimators = 10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
final_metrics = accuracy_score(y_test, y_pred)
print(final_metrics)

final_metrics1 = confusion_matrix(y_test, y_pred)
print(final_metrics1)

cross_val_Scoring = cross_val_score(X,y,cv=5,scoring="accuracy")
print(cross_val_Scoring)
