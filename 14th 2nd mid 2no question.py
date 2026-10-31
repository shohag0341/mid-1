
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Create dataset
data = {
    'Advertising_Cost': [5, 10, 15, 20, 25],
    'Customer_Complaints': [2, 4, 7, 12, 18]
}

df = pd.DataFrame(data)

# Define variables
X = sm.add_constant(df['Advertising_Cost'])
y = df['Customer_Complaints']

# Fit Poisson Regression
model = sm.GLM(y, X, family=sm.families.Poisson())
result = model.fit()

# Display regression results
print(result.summary())

# Expected complaints at X = 10
pred_10 = result.predict([1, 10])[0]
print("Expected complaints at 10:", pred_10)

# Expected complaints at X = 20
pred_20 = result.predict([1, 20])[0]
print("Expected complaints at 20:", pred_20)

# Calculate IRR
beta = result.params['Advertising_Cost']
IRR = np.exp(beta)
print("IRR:", IRR)

# Interpret coefficient
print("Percentage change:", (IRR - 1) * 100)
