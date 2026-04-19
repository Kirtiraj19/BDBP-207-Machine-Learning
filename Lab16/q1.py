# Write a Python program to aggregate  predictions from multiple trees to output a final prediction for a regression problem.

import numpy as np

# Predictions from 5 trees for one sample
tree_predictions = [150, 160, 155, 158, 152]

# Final prediction = average
final_prediction = np.mean(tree_predictions)

print("Predictions from Trees:", tree_predictions)
print("Final Prediction:", final_prediction)