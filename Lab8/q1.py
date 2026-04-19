#L2 norm and #L1 norm from scratch

import numpy as np
X= [[0.04552144],
 [2.01493552],
 [1.01211857],
 [3.87520543],
 [4.09000681],
 [0.02109608]]

X = np.array(X)   # convert list to numpy array

def L1_norm(X, lamda):
    return lamda * np.sum(np.abs(X))   # take absolute values before sum
L1 = L1_norm(X, lamda=0.01)
print("L1 norm =", L1)


def L2_norm(X, lamda):
    return lamda * np.sqrt(np.sum(X**2))   # square → sum → sqrt
L2 = L2_norm(X, lamda=0.01)
print("L2 norm =", L2)
