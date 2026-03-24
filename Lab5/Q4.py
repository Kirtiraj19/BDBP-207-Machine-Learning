# Implement logistic regression using scikit-learn for the breast cancer dataset

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
data = load_breast_cancer()

# separate features and labels
X = data.data
y = data.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create model
model = LogisticRegression(max_iter=10000)

# train model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
