import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Load the data
df = pd.read_csv('Mall_Customers.csv')

# Step 1: Basic Exploration
print(df.head())
print(df.info())

# Step 2: Encode categorical column 'Gender'
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male:1, Female:0

# Step 3: Check correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="magma")
plt.title("Correlation Heatmap")
plt.show()

# Step 4: Feature and Target Split
X = df.drop(columns=['CustomerID'])  # Drop ID
y = df['Spending Score (1-100)']  # (Optional for supervised FS)
#
# # Step 5: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#
# # Step 6: Feature Selection using SelectKBest (ANOVA F-test)
selector = SelectKBest(score_func=f_classif, k=1)
fit = selector.fit(X_scaled, y)
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': fit.scores_
}).sort_values(by='Score', ascending=False)
print("\nSelectKBest Feature Scores:")
print(feature_scores)
#
# # Step 7: Feature Importance using RandomForest
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_scaled, y)
# importances = clf.feature_importances_
#
# # Plot feature importances
# feat_importances = pd.Series(importances, index=X.columns)
# feat_importances.sort_values().plot(kind='barh', title='Feature Importances (Random Forest)')
# plt.show()
