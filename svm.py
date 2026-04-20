
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data      # Features (sepal length, sepal width, etc.)
y = iris.target    # Labels (flower types)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = svm.SVC(kernel='linear')   # 'linear' kernel is easiest for beginners

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
