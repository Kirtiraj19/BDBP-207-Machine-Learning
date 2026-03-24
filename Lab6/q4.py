#Use validation set to do feature and model selection.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# load dataset
data = load_breast_cancer()

# features and labels
X = data.data
y = data.target

# Step 1: split into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Selection (select top 5 features)
selector = SelectKBest(score_func=chi2, k=5)

# fit selector only on training data
X_train_selected = selector.fit_transform(X_train, y_train)

# apply same transformation on validation data
X_val_selected = selector.transform(X_val)

# Step 3: Model 1 - Logistic Regression
model1 = LogisticRegression(max_iter=10000)

# train model
model1.fit(X_train_selected, y_train)

# evaluate on validation set
acc1 = model1.score(X_val_selected, y_val)

# Step 4: Model 2 - Decision Tree
model2 = DecisionTreeClassifier()

# train model
model2.fit(X_train_selected, y_train)

# evaluate on validation set
acc2 = model2.score(X_val_selected, y_val)

# Step 5: Compare models
print("Logistic Regression Accuracy:", acc1)
print("Decision Tree Accuracy:", acc2)

# Step 6: Select best model
if acc1 > acc2:
    print("Best Model: Logistic Regression")
else:
    print("Best Model: Decision Tree")