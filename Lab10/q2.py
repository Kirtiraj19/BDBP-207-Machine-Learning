# Implement information gain measures. The function should accept data points for parents,
# data points for both children and return an information gain value.

import math
import pandas as pd

df = pd.read_csv("Entropy.csv")
# target column
y = df.iloc[:, -1]

# entropy function
def entropy(y):
    probs = y.value_counts(normalize=True)

    e = 0
    for p in probs:
        e += (-p) * math.log2(p)
    return e

# information gain function
def information_gain(df, feature, target):
    total_entropy = entropy(df[target])

    values = df[feature].unique()

    weighted_entropy = 0

    for val in values:
        subset = df[df[feature] == val]
        weight = len(subset) / len(df)
        weighted_entropy += weight * entropy(subset[target])

    ig = total_entropy - weighted_entropy
    return ig

# calculate for each feature
print("Entropy =", entropy(df["play"]))
print("IG(weather) =", information_gain(df, "weather", "play"))
print("IG(windy) =", information_gain(df, "windy", "play"))