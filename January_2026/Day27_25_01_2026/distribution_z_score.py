import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Sample data
data = [72, 75, 68, 80, 85, 65, 76, 79, 82, 71]

# Calculate z-scores
mean = np.mean(data)
std_dev = np.std(data, ddof=0)  # Population standard deviation
z_scores = [(x - mean) / std_dev for x in data]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the standard normal curve (z-distribution)
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)
plt.plot(x, y, 'b-', lw=2, label='Standard Normal (Z) Distribution')

# Add vertical lines for each z-score
for i, z in enumerate(z_scores):
    plt.axvline(x=z, color='r', alpha=0.3, linestyle='--')
    plt.text(z, 0.01, f'z={z:.2f}', rotation=90)

# Highlight the data points on the curve
for z in z_scores:
    plt.plot(z, stats.norm.pdf(z, 0, 1), 'ro')

# Add shaded regions for standard deviations
colors = ['#e6f7ff', '#cceeff', '#b3e6ff']
for i in range(1, 4):
    plt.fill_between([-i, i], 0, stats.norm.pdf(0, 0, 1), alpha=0.2, color=colors[i-1],
                    label=f'Within {i} std dev ({stats.norm.cdf(i) - stats.norm.cdf(-i):.1%})')

plt.title('Z-Distribution with Z-Scores from Our Example')
plt.xlabel('Z-Score (Standard Deviations from Mean)')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.3)
plt.legend()

# Print summary statistics
print("Z-Score Example:")
print(f"Original data: {data}")
print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Z-scores: {[round(z, 2) for z in z_scores]}")
print(f"\nPercentage of data within:")
print(f"±1 standard deviation: {stats.norm.cdf(1) - stats.norm.cdf(-1):.2%}")
print(f"±2 standard deviations: {stats.norm.cdf(2) - stats.norm.cdf(-2):.2%}")
print(f"±3 standard deviations: {stats.norm.cdf(3) - stats.norm.cdf(-3):.2%}")

plt.tight_layout()
plt.show()