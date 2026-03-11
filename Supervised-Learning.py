import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[1], [2], [3], [4], [5]])   
y = np.array([35, 40, 50, 55, 65])      

model = LinearRegression()

model.fit(X, y)

hours = np.array([[6]])
prediction = model.predict(hours)

print("Predicted Marks for 6 hours study:", prediction[0])
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Supervised Learning - Linear Regression")
plt.show()
