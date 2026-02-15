from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Binomial distribution
n_trials = 10
p_success = 0.5

# Generate population from binomial distribution
population = binom.rvs(n=n_trials, p=p_success, size=100000)
sample_means = [np.mean(np.random.choice(population, size=50)) for _ in range(1000)]

# Plot
plt.hist(sample_means, bins=30, density=True, alpha=0.6)
plt.title('Sample Means - Binomial Population')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.show()
