import numpy as np
from scipy.stats import chi2_contingency

table = np.array([
    [40,35,25],
    [30,45,25],
    [20,30,50]
])

chi2, p ,df , observed = chi2_contingency(
    table
)
print("chisquare=",chi2)
print("pvalue=",p)
print("degrees of freedom=",df)
print("expected values:\n",observed)

if p<0.05:
    print("We reject the null hypothesis. theatment are not equal")
else:
    print("Failed to reject the null hypothesis. treatment are same")
