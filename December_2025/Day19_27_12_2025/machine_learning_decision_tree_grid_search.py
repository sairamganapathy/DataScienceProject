import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

column_names = ['buying_price', 'maintenance_cost', 'doors_numbers',
                'person_number', 'luggage_space', 'safety', 'class']
df = pd.read_csv("car_evaluation.csv", names=column_names, header=None)
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])
X = df.drop("class", axis=1)
y = df['class']
encoder = ce.OrdinalEncoder(cols=['buying_price', 'maintenance_cost', 'doors_numbers',
                                  'person_number', 'luggage_space', 'safety'])
X_encoded = encoder.fit_transform(X)

dt_model = DecisionTreeClassifier(random_state=42)
param_grid = {'criterion': ['gini','entropy'],
              'max_depth': [2,3,4,5,6, None],
              'min_samples_split': [2,5,10],
              'min_samples_leaf': [1,2,4]}
grid_search = GridSearchCV(estimator=dt_model,
                           cv = 5,
                           param_grid=param_grid,
                           scoring='accuracy',
                           n_jobs=-1)
grid_search.fit(X_encoded, y)
cross_val_score_evaluation = cross_val_score(dt_model,X_encoded,y,cv=5,scoring="accuracy")
print(cross_val_score_evaluation)
print("best params: ", grid_search.best_params_)
print("best cross validation score: ", grid_search.best_score_)
best_model = grid_search.best_estimator_
print("best estimator: ", grid_search.best_estimator_)
y_pred = best_model.predict(X_encoded)
from sklearn.metrics import accuracy_score
print("Accuracy on full training set:", accuracy_score(y, y_pred))

