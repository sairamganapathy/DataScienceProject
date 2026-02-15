import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = [72, 75, 68, 80, 85, 65, 76, 79, 82, 71]

# Calculate z-scores
mean = np.mean(data)
std_dev = np.std(data, ddof=0)  # population standard deviation(delta degree of freedom)
z_scores = [(mohan - mean) / std_dev for mohan in data] #list comprehension (executing for loop inside the list)
print(z_scores)

rounded_z_score = [round(pooja,2) for pooja in z_scores]

print("Data:", data)
print("Mean:", mean)
print("Standard Deviation:", std_dev)
print([round(z, 2) for z in z_scores])

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(range(len(data)), z_scores)
plt.axhline(y=0, color='r', linestyle='-')
plt.axhline(y=1, color='g', linestyle='--')
plt.axhline(y=-1, color='g', linestyle='--')
plt.title('Z-Scores for Each Data Point')
plt.xlabel('Data Point Index')
plt.ylabel('Z-Score')
plt.show()