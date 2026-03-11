
import pandas as pd

data = {
    "Name": ["Aman", "Riya", "Rahul", "Neha"],
    "Gender": ["Male", "Female", "Male", "Female"],
    "Age": [23, 21, 25, 22]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

dummy = pd.get_dummies(df["Gender"])

print("\nDummy Variables:")
print(dummy)

df = pd.concat([df, dummy], axis=1)

print("\nFinal Dataset with Dummy Variables:")
print(df)
