import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate Uniform Population
population = np.random.uniform(0, 1, 100000)
sample_means = [np.mean(np.random.choice(population, size=40)) for _ in range(1000)]

# Plot
sns.histplot(sample_means, bins=30, kde=True)
plt.title('Sample Means - Uniform Population')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.show()
