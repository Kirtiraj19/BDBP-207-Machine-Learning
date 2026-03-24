#Compute the derivative of a sigmoid function and visualize it

import numpy as np
import matplotlib.pyplot as plt

# sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# derivative of sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)

# input values
x = np.linspace(-10,10,100)

# derivative values
y = sigmoid_derivative(x)

# plot graph
plt.plot(x,y)

plt.xlabel("x")
plt.ylabel("sigmoid derivative")
plt.title("Derivative of Sigmoid")

plt.show()







# import matplotlib.pyplot as plt
# import math
# x=[i for i in range(1,101)]
# y=[]
# def derivative_sigmoid(z):
#     return (1/(1+math.exp(-z)))*(1-(1/(1+math.exp(-z))))
# for values in x:
#     y.append(derivative_sigmoid(values))
#
# print(y)
# plt.title("derivative sigmoid function")
# plt.plot(x,y)
# plt.show()