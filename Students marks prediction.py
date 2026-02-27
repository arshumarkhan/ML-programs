from sklearn.linear_model import LinearRegression
hours= [[2],[4],[6],[8]]
marks= [60,70,80,90]
model= LinearRegression()
model.fit(hours,marks)
new_hours =[[10]]
predicted_marks = model.predict(new_hours)
print("hours",new_hours[0][0])
print("Predicted marks:",predicted_marks)
