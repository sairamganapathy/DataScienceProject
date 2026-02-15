import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a normal population
np.random.seed(42)
population = np.random.normal(loc=50, scale=10, size=100000)  # mean=50, std=10

# Take repeated samples and calculate means
sample_size = 30
sample_means = [np.mean(np.random.choice(population, size=sample_size)) for _ in range(1000)]

# Plot sample means
sns.histplot(sample_means, bins=30, kde=True)
plt.title('Sample Means - Normal Population')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.show()
