import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Sample data (small samples)
group1 = [72, 75, 68, 80, 85, 65, 76]
group2 = [62, 68, 70, 75, 71, 60, 65]

# Calculate t-test
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
df = len(group1) + len(group2) - 2  # Degrees of freedom

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the t-distribution for different degrees of freedom
x = np.linspace(-4, 4, 1000)
y_t = stats.t.pdf(x, df)  # t-distribution with our example's df
y_t1 = stats.t.pdf(x, 1)  # t with 1 degree of freedom (very heavy tails)
y_t5 = stats.t.pdf(x, 5)  # t with 5 degrees of freedom
y_t30 = stats.t.pdf(x, 30)  # t with 30 degrees of freedom (close to normal)
y_normal = stats.norm.pdf(x, 0, 1)  # standard normal for comparison

plt.plot(x, y_normal, 'b-', lw=2, label='Z-distribution (Normal)')
plt.plot(x, y_t, 'g-', lw=2, label=f't-distribution (df={df})')
plt.plot(x, y_t1, 'r--', lw=1, label='t-distribution (df=1)')
plt.plot(x, y_t5, 'm--', lw=1, label='t-distribution (df=5)')
plt.plot(x, y_t30, 'c--', lw=1, label='t-distribution (df=30)')

# Mark our t-statistic
plt.axvline(x=t_stat, color='r', linestyle='-')
plt.text(t_stat, 0.05, f't={t_stat:.2f}', rotation=90, color='r')

# Shade the rejection regions (two-tailed test at α=0.05)
critical_t = stats.t.ppf(0.975, df)  # Critical value for α=0.05 (two-tailed)
plt.axvline(x=critical_t, color='k', linestyle=':', label=f'Critical t=±{critical_t:.2f}')
plt.axvline(x=-critical_t, color='k', linestyle=':')

# Shade rejection regions
x_fill_right = np.linspace(critical_t, 4, 100)
y_fill_right = stats.t.pdf(x_fill_right, df)
plt.fill_between(x_fill_right, y_fill_right, alpha=0.3, color='red')

x_fill_left = np.linspace(-4, -critical_t, 100)
y_fill_left = stats.t.pdf(x_fill_left, df)
plt.fill_between(x_fill_left, y_fill_left, alpha=0.3, color='red', label='Rejection Regions (α=0.05)')

plt.title('T-Distribution Comparison with Different Degrees of Freedom')
plt.xlabel('T-Value')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.3)
plt.legend()

# Print summary statistics
print("T-Test Example:")
print(f"Group 1: {group1}")
print(f"   Mean: {np.mean(group1):.2f}")
print(f"   Standard Deviation: {np.std(group1, ddof=1):.2f}")
print(f"Group 2: {group2}")
print(f"   Mean: {np.mean(group2):.2f}")
print(f"   Standard Deviation: {np.std(group2, ddof=1):.2f}")
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {df}")
print(f"Critical t-value (α=0.05, two-tailed): ±{critical_t:.2f}")
print(f"Result: {'Reject null hypothesis (means are different)' if p_value < 0.05 else 'Fail to reject null hypothesis'}")

plt.tight_layout()
plt.show()