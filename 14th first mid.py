
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Create dataset
data = {
    'Exercise_Hours': [1, 2, 3, 4, 5],
    'Doctor_Visits': [5, 4, 3, 2, 1]
}

df = pd.DataFrame(data)

# Independent and dependent variables
X = sm.add_constant(df['Exercise_Hours'])
y = df['Doctor_Visits']

# Fit Poisson Regression
model = sm.GLM(y, X, family=sm.families.Poisson())
result = model.fit()

# Display coefficients
print(result.summary())

# Question 1: Expected visits at 2 hours
pred_2 = result.predict([1, 2])[0]
print("Expected visits at 2 hours:", pred_2)

# Question 2: Expected visits at 4 hours
pred_4 = result.predict([1, 4])[0]
print("Expected visits at 4 hours:", pred_4)

# Question 3: Incidence Rate Ratio
IRR = np.exp(result.params['Exercise_Hours'])
print("IRR:", IRR)

# Question 4: Interpret coefficient
beta = result.params['Exercise_Hours']
print("Exercise coefficient:", beta)
print("Percentage change:", (IRR - 1) * 100)
