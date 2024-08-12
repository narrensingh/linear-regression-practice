import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normalisation import normalization

np.random.seed(42)
x = np.random.rand(100,1)*2
y = 4 + 3*x + np.random.rand(100,1)
x_b = np.c_[np.ones((x.shape[0],1)),x]

def predict(x_b,params):
    return x_b.dot(params)

def cost(error):
    cost = (np.mean(error**2))/2
    return cost

def gradient_descend(x_b,y,n_iterations=1000,learning_rate=0.01):
    params = np.zeros((2,1))
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        predictions = predict(x_b,params)
        error = predictions - y 
        params[0] -= learning_rate*np.mean(error)
        params[1] -= learning_rate*np.dot(error.T,x_b[:,1])
        e = cost(error)
        cost_history[i] = e
        if i%100 == 0:
            print(f'The Cost In {i}th iteration is: {e}')
    return params,cost_history

def predictions(x_b,params):
    predictions = x_b.dot(params)
    return predictions

params,cost_history = gradient_descend(x_b,y,n_iterations=1000,learning_rate=0.01)
model = predictions(x_b,params).reshape(100)
plt.subplot(2,1,1)
plt.plot(cost_history)
plt.xlabel('# iterations')
plt.ylabel('Cost')
plt.title('Learning Curve')
plt.subplot(2,1,2)
plt.scatter(x,y,alpha=0.5)
plt.plot(x,model,color='red')
plt.title('Model')
plt.show()