# Perform 10-fold cross validation for SONAR dataset in scikit-learn using
# logistic regression. SONAR dataset is a binary classification problem with target
# variables as Metal or Rock. i.e. signals are from metal or rock.

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('sonar data.csv', header=None)

X = data.iloc[:, :-1]    #all columns except features
y = data.iloc[:, -1]    #target column 'R' or 'M'

# Machine learning models cannot handle text labels directly
# So we convert: R (Rock) → 0  and  M (Metal) → 1
encoder = LabelEncoder()
y = encoder.fit_transform(y)
#Logistic Regression is used for binary classification
model = LogisticRegression(max_iter=1000)
# n_splits = 10 - dataset divided into 10 parts
# shuffle = True - randomize data before splitting
# random_state - ensures same result every time
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
#cross_val_score will split data into 10 folds, train model on 9 folds and test on remaining fold
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print("Accuracy for each fold:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())
