
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [2, 6],
    [3, 7],
    [4, 8],
    [5, 6],
    [6, 7]
])

y = np.array([50, 55, 65, 70, 75])

model = LinearRegression()

model.fit(X, y)

new_data = np.array([[4, 7]])
prediction = model.predict(new_data)

print("Predicted Marks:", prediction[0])
