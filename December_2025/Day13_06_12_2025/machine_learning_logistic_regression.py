import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = {"study_hours":[1,2,3,4,5,6,7,8,9,10],
        "test_score":[35,40,45,50,55,65,70,75,80,90],
        "result":[0,0,0,0,0,1,1,1,1,1]
        }
df = pd.DataFrame(data)
x = df[['study_hours','test_score']]
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
model = LogisticRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
print("classification report",classification_report(y_test,y_predict))
print("\n accuracy score",accuracy_score(y_test,y_predict))
new_student = [[5,60]]
prediction = model.predict(new_student)
print("\n prediction for new student: ","pass" if prediction[0]==1 else "fail")

confusion_matrix = confusion_matrix(y_test,y_predict)
print(confusion_matrix)