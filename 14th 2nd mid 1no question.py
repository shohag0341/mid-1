
import numpy as np
from scipy.stats import chi2_contingency

# Create contingency table
data = np.array([
    [60, 40],
    [30, 70]
])

# Calculate Odds Ratio
a, b = data[0]
c, d = data[1]

OR = (a * d) / (b * c)
print("Odds Ratio:", OR)

# Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(
    data, correction=False
)

print("Chi-square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)

print("Expected Frequencies:")
print(expected)

# Decision at 5% significance level
if p < 0.05:
    print("Reject Null Hypothesis")
    print("Smoking and Lung Disease are associated.")
else:
    print("Fail to Reject Null Hypothesis")
    print("Smoking and Lung Disease are independent.")
  
