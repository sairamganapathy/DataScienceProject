import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Sample data for ANOVA
group1 = [72, 75, 68, 80, 85, 65, 76]
group2 = [62, 68, 70, 75, 71, 60, 65]
group3 = [80, 82, 84, 77, 81, 79, 83]

# Calculate F-statistic
f_stat, p_value = stats.f_oneway(group1, group2, group3)
df1 = 3 - 1  # Between groups degrees of freedom (k-1)
df2 = len(group1) + len(group2) + len(group3) - 3  # Within groups degrees of freedom (N-k)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot F-distributions with different degrees of freedom
x = np.linspace(0, 6, 1000)  # F-distribution is only defined for positive values
y_f = stats.f.pdf(x, df1, df2)  # Our example's F distribution
y_f_1_10 = stats.f.pdf(x, 1, 10)  # Example with df1=1, df2=10
y_f_5_20 = stats.f.pdf(x, 5, 20)  # Example with df1=5, df2=20
y_f_10_5 = stats.f.pdf(x, 10, 5)  # Example with df1=10, df2=5

plt.plot(x, y_f, 'r-', lw=2, label=f'F-distribution (df1={df1}, df2={df2})')
plt.plot(x, y_f_1_10, 'b--', lw=1, label='F-distribution (df1=1, df2=10)')
plt.plot(x, y_f_5_20, 'g--', lw=1, label='F-distribution (df1=5, df2=20)')
plt.plot(x, y_f_10_5, 'm--', lw=1, label='F-distribution (df1=10, df2=5)')

# Mark our F-statistic
plt.axvline(x=f_stat, color='g', linestyle='-')
plt.text(f_stat, 0.1, f'F={f_stat:.2f}', rotation=90, color='g')

# Shade the rejection region (α=0.05)
critical_f = stats.f.ppf(0.95, df1, df2)  # Critical value for α=0.05
plt.axvline(x=critical_f, color='k', linestyle=':', label=f'Critical F={critical_f:.2f}')

x_fill = np.linspace(critical_f, 6, 100)
y_fill = stats.f.pdf(x_fill, df1, df2)
plt.fill_between(x_fill, y_fill, alpha=0.3, color='red', label='Rejection Region (α=0.05)')

plt.title('F-Distribution Comparison with Different Degrees of Freedom')
plt.xlabel('F-Value')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.3)
plt.legend()

# Print group statistics
print("ANOVA F-Test Example:")
groups = [group1, group2, group3]
for i, group in enumerate(groups, 1):
    print(f"Group {i}: {group}")
    print(f"   Mean: {np.mean(group):.2f}")
    print(f"   Variance: {np.var(group, ddof=1):.2f}")

# Calculate between-group and within-group variances
all_data = group1 + group2 + group3
grand_mean = np.mean(all_data)
group_means = [np.mean(group) for group in groups]
n = len(all_data)
k = len(groups)

# Between-group sum of squares
ss_between = sum(len(group) * (mean - grand_mean)**2 for group, mean in zip(groups, group_means))

# Within-group sum of squares
ss_within = sum(sum((x - group_mean)**2 for x in group) for group, group_mean in zip(groups, group_means))

# Mean squares
ms_between = ss_between / df1
ms_within = ss_within / df2

# F-statistic calculation
f_calc = ms_between / ms_within

print("\nANOVA Calculation:")
print(f"Grand Mean: {grand_mean:.2f}")
print(f"Between-group sum of squares: {ss_between:.2f}")
print(f"Within-group sum of squares: {ss_within:.2f}")
print(f"Mean square between: {ms_between:.2f}")
print(f"Mean square within: {ms_within:.2f}")
print(f"\nF-statistic: {f_calc:.2f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: df1={df1}, df2={df2}")
print(f"Critical F-value (α=0.05): {critical_f:.2f}")
print(f"Result: {'Reject null hypothesis (means are different)' if p_value < 0.05 else 'Fail to reject null hypothesis'}")

plt.tight_layout()
plt.show()