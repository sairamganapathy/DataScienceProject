import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Sample data for ANOVA (three groups)
group1 = [72, 75, 68, 80, 85, 65, 76]
group2 = [62, 68, 70, 75, 71, 60, 65]
group3 = [80, 82, 84, 77, 81, 79, 83]

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)

print("Group 1:", group1)
print("Group 2:", group2)
print("Group 3:", group3)
print("F-statistic:", round(f_stat, 2))
print("P-value:", round(p_value, 4))
print("Significant difference" if p_value < 0.05 else "No significant difference")

# Visualization
plt.figure(figsize=(10, 6))
plt.boxplot([group1, group2, group3])
plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])
plt.title(f'ANOVA: F={round(f_stat, 2)}, p={round(p_value, 4)}')

"""below code is older school of assigning numeric values inside the string dynamically"""
####plt.title('ANOVA: F="%d", p="%d"'%(round(f_stat, 2),round(p_value, 4)))
plt.ylabel('Values')
plt.show()