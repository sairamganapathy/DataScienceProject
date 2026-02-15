import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

data = pd.read_csv("Mall_Customers.csv")

X = data[["Annual Income (k$)","Spending Score (1-100)"]].values

X_Scaled = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.5, min_samples=5).fit(X_Scaled)
labels = db.labels_

plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c=labels, cmap= "plasma", s=50, alpha=0.7)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN clustering on Mall Customers")
plt.colorbar(label="Cluster Label")
plt.show()
