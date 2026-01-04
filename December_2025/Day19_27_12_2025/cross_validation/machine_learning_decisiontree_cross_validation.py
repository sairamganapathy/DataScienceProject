import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

column_names = ['buying_price', 'maintenance_cost', 'doors_numbers',
                'person_number', 'luggage_space', 'safety', 'class']
df = pd.read_csv("../car_evaluation.csv", names=column_names, header=None)

label_encoder = LabelEncoder()
# print(df.head())
df['class'] = label_encoder.fit_transform(df['class'])
# print(df.head())
X = df.drop("class", axis=1)
# print(X.head())
encoder = ce.OrdinalEncoder(cols=['buying_price', 'maintenance_cost', 'doors_numbers',
                                  'person_number', 'luggage_space', 'safety'])
X_encoded = encoder.fit_transform(X)
# print(X_encoded.head())

x = X_encoded
y = df['class']

model = DecisionTreeClassifier()

# cross_val_score_evaluation = cross_val_score(model,x,y,cv=5,scoring="accuracy")
cross_val_score_evaluation = cross_val_score(model,x,y,cv=5,scoring="accuracy")
print(cross_val_score_evaluation)