import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('housing.csv')
x = data[['median_income']].values
y = data['median_house_value'].values #column matrix
ones = np.ones((x.shape))
x = np.hstack((ones,x))
def normaleq(x,y):
    return np.linalg.inv(x.T @ x) @ x.T @ y
param = normaleq(x,y)
print(len(x))
def pridict(x,param):
    return x@param

y_pred = pridict(x,param)
print(len(y_pred))

mse = np.mean((y_pred-y)**2)
print(mse)

# Plot the data and the regression line
plt.scatter(data['median_income'], data['median_house_value'], color='blue', label='Original data')
plt.plot(data['median_income'], y_pred, color='red', label='Fitted line')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend()
plt.show()