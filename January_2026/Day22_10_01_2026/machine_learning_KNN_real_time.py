import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
df = pd.read_csv(url, names = names, delim_whitespace = True)
df.drop("car_name", axis = 1, inplace = True)

df.replace("?", np.nan, inplace = True)
df["horsepower"] = pd.to_numeric(df["horsepower"])

# Drop any rows with NaN values
df.dropna(inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('mpg', axis=1), df['mpg'], test_size=0.2, random_state=42)

# Create a k-NN regression model. Because we are using continuous data going with regressor
knn = KNeighborsRegressor()

# Fit the model to the training data
knn.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
#as it is regressor using mean_squared_error or can use R2_score only.
#Only for classifier accuracy, precision_score be used
print(f"Test MSE: {mse:.3f}")