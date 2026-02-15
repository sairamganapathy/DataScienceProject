import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate Exponential Population
population = np.random.exponential(scale=2, size=100000)
sample_means = []

# Draw 1000 samples, each with sample_size elements
sample_size = 30
for _ in range(1000):
    sample = np.random.choice(population, size=sample_size)
    sample_means.append(np.mean(sample))

# Plot
sns.histplot(sample_means, bins=30, kde=True)
plt.title('Sample Means - Exponential Population')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.show()
