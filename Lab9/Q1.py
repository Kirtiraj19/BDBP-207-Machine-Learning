# Write a program to partition a dataset (simulated data for regression)  into two parts,
# based on a feature (BP) and for a threshold, t = 80. Generate additional two partitioned datasets
# based on different threshold values of t = [78, 82].

import pandas as pd
import numpy as np

# Step 1: Create simulated dataset
np.random.seed(0)

data = pd.DataFrame({
    'BP': np.random.randint(60, 100, 20),  # Blood Pressure feature
    'Target': np.random.randint(100, 200, 20)  # Regression target
})

print("Original Dataset:\n", data)


# Function to partition dataset
def partition(data, threshold):
    left = data[data['BP'] <= threshold]
    right = data[data['BP'] > threshold]

    print(f"\nThreshold = {threshold}")
    print("Left (BP <= t):\n", left)
    print("Right (BP > t):\n", right)


# Step 2: Apply thresholds
partition(data, 80)
partition(data, 78)
partition(data, 82)