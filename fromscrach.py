import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normalisation import normalization

np.random.seed(42)
x = np.random.rand(100,1)*2
y = 4 + 3*x + np.random.rand(100,1)

def predict(x,param):
    return x.dot(param)

def cost_function(x,y,param):
    predictions = predict(x,param)
    cost = (((y - predictions)**2).mean())/2
    return cost

def gradient_descend(x,y,param,learning_rate,num_interations):
    m = y.shape[0]
    cost_history = np.zeros(num_interations)
    for i in range(num_interations):
        predictions = predict(x,param)
        error = predictions - y
        gradient_weight = ((x.T.dot(error)).sum())/m
        gradient_bias = error.mean()
        param[1] -= learning_rate*gradient_weight
        param[0] -= learning_rate*gradient_bias
        cost_history[i] = cost_function(x,y,param)
    return param,cost_history

ones = np.ones((100,1))
x_b = np.column_stack((ones,x))
theta = np.random.rand(2,1)
learning_rate = 0.01
num_iterations = 1000
theta, cost_history = gradient_descend(x_b, y, theta, learning_rate, num_iterations)

plt.subplot(2,1,1)
plt.scatter(x,y,alpha=0.5)
plt.plot(x,predict(x_b,theta),color='red')
plt.title('Comparison')
plt.subplot(2,1,2)
plt.plot(cost_history)
plt.title('Cost function')
plt.suptitle('Linear regression model implemented from scratch')
plt.show()