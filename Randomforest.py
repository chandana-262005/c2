import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Example dataset (replace with your actual data)
X = np.array([[30], [40], [50], [60], [70], [80], [90]])
y = np.array([35, 45, 55, 65, 75, 85, 95])

# Create model
RandomForestRegModel = RandomForestRegressor()

# Train model
RandomForestRegModel.fit(X, y)

# Predict for 70 marks
X_marks = [[70]]
prediction = RandomForestRegModel.predict(X_marks)

print("Predicted value:", prediction)
