#Implement gradient descent algorithm from scratch using Python


import numpy as np

# STEP 1 : create training data

# input feature values
x = np.array([1, 2, 3, 4])

# true output values
y = np.array([2, 4, 6, 8])

# number of data points
n = len(x)

# STEP 2 : initialize parameters

# weight (slope) initial value
w = 0

# bias initial value
b = 0

# learning rate (step size)
learning_rate = 0.01

# number of iterations
iterations = 1000

# STEP 3 : gradient descent loop

for i in range(iterations):

    # predicted output using current model
    # formula: y_pred = wx + b
    y_pred = w * x + b

    # calculate error
    error = y_pred - y

    # compute gradients

    # gradient of cost with respect to w
    # derivative of MSE wrt w
    dw = (2/n) * np.sum(error * x)

    # gradient wrt bias
    db = (2/n) * np.sum(error)

    # update parameters

    w = w - learning_rate * dw
    b = b - learning_rate * db

# final learned parameters

print("Learned weight:", w)
print("Learned bias:", b)




#this is common
# import numpy as np
#
# def gradient_descent(X_with_bias_train,y_np_train,h,alpha,theta_matrix):
#     thet=[]
#     for i in range(len(X_with_bias_train[0])):
#         s=0
#         for j in range(len(y_np_train)):
#             s+=(h[j][0]-y_np_train[j][0])*X_with_bias_train[j][i]    # summation
#
#         thet.append(theta_matrix[i][0]-(alpha*s))        # multiply by alpha and subtract with theta
#
#     theta=np.array(thet)
#     return theta.reshape(-1,1)




