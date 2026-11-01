import pandas as pd
import statsmodels.api as sm

# Create example dataset
data = pd.DataFrame({
    'Diabetes': [0, 1, 0, 1, 1, 0, 1, 0],
    'Age': [25, 45, 30, 55, 60, 35, 50, 20]
})

# Define dependent and independent variables
X = data[['Age']]
y = data['Diabetes']

# Add intercept
X = sm.add_constant(X)

# Fit logistic regression
model = sm.GLM(
    y,
    X,
    family=sm.families.Binomial(
        link=sm.families.links.Logit()
    )
)

result = model.fit()

# Display results
print(result.summary())

# Calculate odds ratios
odds_ratios = pd.DataFrame({
    'Coefficient': result.params,
    'Odds Ratio': result.params.apply(lambda x: __import__('numpy').exp(x))
})

print("\nOdds Ratios:")
print(odds_ratios)


import matplotlib.pyplot as plt
import numpy as np

# Generate age values
age_range = np.linspace(20, 70, 100)

# Predict diabetes probability
X_new = pd.DataFrame({'const': 1, 'Age': age_range})
predicted = result.predict(X_new)

plt.plot(age_range, predicted)
plt.xlabel("Age")
plt.ylabel("Predicted probability")
plt.title("Logistic Regression")
plt.ylim(0, 1)
plt.grid()
plt.show()
