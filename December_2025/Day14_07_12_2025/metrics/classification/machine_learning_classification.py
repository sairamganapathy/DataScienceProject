import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1️⃣ Create a simple DataFrame
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Test_Score": [35, 40, 45, 50, 55, 65, 70, 75, 85, 90],
    "Result": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]   # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
print(df)

# 2️⃣ Split features and target
X = df[["Study_Hours", "Test_Score"]]
y = df["Result"]

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Create & train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5️⃣ Predictions
y_pred = model.predict(X_test)
print(y_pred)

print(accuracy_score(y_test, y_pred))

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix_evaluation = confusion_matrix(y_test, y_pred)
print(confusion_matrix_evaluation)



