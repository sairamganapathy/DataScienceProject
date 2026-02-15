import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Sample data (small sample)
group1 = [72, 75, 68, 80, 85, 65, 76]
group2 = [62, 68, 70, 75, 71, 60, 65]

# Calculate t-test
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)

print("Group 1:", group1)
print("Group 2:", group2)
print("T-statistic:", round(t_stat, 2))
print("P-value:", round(p_value, 4))
print("Significant difference:" if p_value < 0.05 else "No significant difference:")

# Visualization
plt.figure(figsize=(10, 6))
plt.boxplot([group1, group2])
plt.xticks([1, 2], ['Group 1', 'Group 2'])
plt.title(f'T-test: t={round(t_stat, 2)}, p={round(p_value, 4)}')
plt.ylabel('Values')
plt.show()