'''Use the above simulated CSV file and implement the following from scratch in Python
Read simulated data csv file
Form x and y (disease_score_fluct)
Write a function to compute hypothesis
Write a function to compute the cost
Write a function to compute the derivative
Write update parameters logic in the main function'''

#read csv file
import pandas as pd
import numpy as np
from scipy.special.cython_special import huber
import math
df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')

# X and y - disease_score_fluct
X = df.iloc[:,:5]
y = df['disease_score_fluct']

# function to compute the hypothesis
X_np = X.values
m=X_np.shape[0]
X_with_bias = np.concatenate((np.ones((m,1)), X_np), axis=1)
theta=np.zeros((X_with_bias.shape[1]))
theta_matrix=theta.reshape(6,1)
print("Theta",theta_matrix)
print("x =",X_with_bias)
y_np = y.values.reshape(-1, 1)

print("y =",y_np)

#--step1---set theta value as 0 and compute hypothesis--
def hypothesis(X_with_bias, theta_matrix):
    return np.dot(X_with_bias, theta_matrix)

#--hypothesis is calculated at 1st all hypothesis is 0 --
def cost(h, y_np):
    for i in range(len(y_np)):
        s = 0
        s += ((hypothes[i][0] - y_np[i][0]) ** 2)
    return s*1/2
h=hypothesis(X_with_bias,theta_matrix)
print("h =",h)
print("len h =",len(h[0]))
#print("x",X_with_bias[0][1])
alpha=0.000000924
print("alpha",alpha)
#--Now find theta with alpha and theta and hypothesis--
def find_theta(X_with_bias,y_np,h,alpha,theta_matrix):
    thet=[]
    for i in range(len(X_with_bias[0])):
        s=0
        for j in range(len(y_np)):
            s+=(h[j][0]-y_np[j][0])*X_with_bias[j][i]    # summation

        thet.append(theta_matrix[i][0]-(alpha*s))        # multiply by alpha and subtract with theta

    theta=np.array(thet)
    return theta.reshape(-1,1)


theta=find_theta(X_with_bias,y_np,h,alpha,theta_matrix)
print(theta)
print("-"*30)

for i in range(5):
    Theta=theta
    hypothes = hypothesis(X_with_bias,Theta)
    prev_cost = cost(hypothes, y_np)
    print(i,"cost",prev_cost)
    theta_update = find_theta(X_with_bias,y_np,hypothes,alpha,Theta)
    print(i,"theta_updated",theta_update)
    print(i,"hypothesis",hypothes)
    Theta=theta_update




# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score           #, mean_absolute_error, mean_squared_error
#
# #read the data
# def load_data():
#     df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
#     print("Dataset shape:", df.shape)
#     print(df.head())
#     print(df.describe())
#
# x = load_data['x'].values
# y = load_data['disease_score_fluct'].values
#
# # 2. Hypothesis function
# def hypothesis(x, theta0, theta1):
#     return theta0 + theta1 * x
#
# # 3. Cost function (Mean Squared Error)
# def compute_cost(x, y, theta0, theta1):
#     m = len(y)
#     y_pred = hypothesis(x, theta0, theta1)
#     cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
#     return cost
#
# # 4. Gradient calculation
# def compute_gradients(x, y, theta0, theta1):
#     m = len(y)
#     y_pred = hypothesis(x, theta0, theta1)
#
#     d_theta0 = (1 / m) * np.sum(y_pred - y)
#     d_theta1 = (1 / m) * np.sum((y_pred - y) * x)
#
#     return d_theta0, d_theta1
#
# # 5. Gradient Descent algorithm
# def gradient_descent(x, y, alpha, iterations):
#     theta0 = 0
#     theta1 = 0
#
#     for i in range(iterations):
#         d0, d1 = compute_gradients(x, y, theta0, theta1)
#         theta0 = theta0 - alpha * d0
#         theta1 = theta1 - alpha * d1
#
#     return theta0, theta1
#
# # 6. Train the model
# learning_rate = 0.01
# iterations = 1000
#
# theta0_gd, theta1_gd = gradient_descent(x, y, learning_rate, iterations)
#
# # 7. Print results
# print("Gradient Descent Results:")
# print("Theta0 (Intercept):", theta0_gd)
# print("Theta1 (Slope):", theta1_gd)











'''for calculating the hypothesis (linear regression)'''
# import numpy as np
# from sklearn.metrics import r2_score
# x = [[1,1,2],
#      [1,2,1],
#      [1,3,3]]
# y = [[3],
#      [4],
#      [5]]
# theta = [[0],
#          [0],
#          [0]]
# def hypothesis(theta,x):
#     hypothesis_one=[]
#     for i in range(len(x[0])):
#         s=0
#         for j in range(len(x)):
#             s=s+theta[j][0]*x[i][j]
#         hypothesis_one.append(s)
#     return np.array(hypothesis_one).reshape(-1,1)
# hypothesis_one = hypothesis(theta,x)
# print('hypothesis (y hat)= ',hypothesis_one)
# alpha=0.001
# def gredient_decent(theta,x,alpha,y):
#     new_theta=[]
#     for i in range(len(x[0])):
#         s=0
#         for j in range(len(y)):
#             s=s+(hypothesis_one[j][0]-y[j][0])*x[j][i]
#         new_theta.append(theta[i][0]-alpha*s)
#     return np.array(new_theta).reshape(-1,1)
# new_theta = gredient_decent(theta,x,alpha,y)
# print('updated theta =',new_theta)
#
# for i in range(35):
#     thet=new_theta
#     prediction=hypothesis(theta,x)
#     print(f'{i}th prediction (Y hat) =',prediction)
#     up_theta= gredient_decent(theta,x,alpha,y)
#     #print(f'{i}th updated theta =',up_theta)
#     theta=up_theta
# r2=r2_score(y,prediction)


