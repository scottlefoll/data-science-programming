#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
 
#%%

#%%
df = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv", header='')
X = df.values[:,1:-1].astype('int')
X = (X - np.mean(X, axis =0)) /  np.std(X, axis = 0)

# X = dwellings_ml.iloc[:, 1:49] 
# X = X.filter(['netprice', 'livearea', 'basement', 'stories'
#               'nocars', 'numbdrm', 'numbaths', 'stories', 
#               'quality_B', 'quality_C', 'condition_AVG', 'quality'])

## Add a bias column to the data
X = np.hstack([np.ones((X.shape[0], 1)),X])
X = MinMaxScaler().fit_transform(X)
Y = df.iloc[:, -1]
Y = np.array(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)


def Sigmoid(z):
    return 1/(1 + np.exp(-z))

def Hypothesis(theta, x):   
    return Sigmoid(x @ theta) 

def Pre_1980_Function(X,Y,theta,m):
    hi = Hypothesis(theta, X)
    _y = Y.reshape(-1, 1)
    J = 1/float(m) * np.sum(-_y * np.log(hi) - (1-_y) * np.log(1-hi))
    return J

def Cost_Function_Derivative(X,Y,theta,m,alpha):
    hi = Hypothesis(theta,X)
    _y = Y.reshape(-1, 1)
    J = alpha/float(m) * X.T @ (hi - _y)
    return J

def Gradient_Descent(X,Y,theta,m,alpha):
    new_theta = theta - Cost_Function_Derivative(X,Y,theta,m,alpha)
    return new_theta

def Accuracy(theta):
    correct = 0
    length = len(X_test)
    prediction = (Hypothesis(theta, X_test) > 0.5)
    _y = Y_test.reshape(-1, 1)
    correct = prediction == _y
    my_accuracy = (np.sum(correct) / length)*100
    print ('LR Accuracy %: ', my_accuracy)

def Logistic_Regression(X,Y,alpha,theta,num_iters):
    m = len(Y)
    for x in range(num_iters):
        new_theta = Gradient_Descent(X,Y,theta,m,alpha)
        theta = new_theta
        if x % 100 == 0:   
            print ('Built Before 1980: ', Pre_1980_Function(X,Y,theta,m))
    Accuracy(theta)


ep = .012

initial_theta = np.random.rand(X_train.shape[1], 1) * 2 * ep - ep
alpha = 0.8
iterations = 2000
Logistic_Regression(X_train, Y_train, alpha, initial_theta, iterations)
#%%