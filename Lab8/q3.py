#ordinal encoding
data = ["Low", "Medium", "High"]

mapping = {"Low": 1, "Medium": 2, "High": 3}
encoded = [mapping[i] for i in data]
print(encoded)

#one-hot encoding
import numpy as np
data = ["Red", "Green", "Blue"]
unique = list(set(data))
one_hot = []
for val in data:
    vector = [1 if val == u else 0 for u in unique]
    one_hot.append(vector)
print(one_hot)

