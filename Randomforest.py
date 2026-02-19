"""
Random Forest Regression Example
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def main():
    """Train and test Random Forest model."""

    # Example dataset
    x = np.array([[30], [40], [50], [60], [70], [80], [90]])
    y = np.array([35, 45, 55, 65, 75, 85, 95])

    # Create model
    model = RandomForestRegressor()

    # Train model
    model.fit(x, y)

    # Predict
    x_marks = [[70]]
    prediction = model.predict(x_marks)

    print("Predicted value:", prediction)


if __name__ == "__main__":
    main()


