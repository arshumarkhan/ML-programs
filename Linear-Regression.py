
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([30, 40, 50, 60, 70])

model = LinearRegression()

model.fit(X, y)


hours = np.array([[6]])
prediction = model.predict(hours)

print("Predicted Marks:", prediction[0])

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression Example")
plt.show()
