import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create a non-normal population (exponential distribution)
np.random.seed(8)
population = np.random.exponential(scale=50, size=100)  # Highly skewed data
print(population)
print(len(population))

sample_size = 30
for _ in range(10):
    print("-------------------------------")
    value = np.random.choice(population,sample_size)
    print(value)

# Step 2: Take multiple random samples & compute their means
sample_size = 30  # Each sample will have 30 data points
num_samples = 1000  # Number of samples we take
#
print("-------------------------------below is mean value-----------------------------------")
sample_means = [np.random.choice(population, sample_size).mean() for _ in range(num_samples)]
print(sample_means)
print(len(sample_means))
#
# # Step 3: Convert to Pandas DataFrame
df = pd.DataFrame({"Sample Mean": sample_means})
#
# # Step 4: Plot the histogram of sample means (CLT effect)
plt.figure(figsize=(8, 5))
sns.histplot(df["Sample Mean"], bins=30, kde=True, color="blue", alpha=0.7)
plt.axvline(np.mean(sample_means), color='red', linestyle='dashed', label="Mean of Sample Means")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.title("Central Limit Theorem: Distribution of Sample Means")
plt.legend()
plt.show()
