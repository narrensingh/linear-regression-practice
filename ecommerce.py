import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error
data = pd.read_csv('customers.csv')
x = data[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = data['Yearly Amount Spent']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
ln = LinearRegression()
ln.fit(x_train.values,y_train.values)
y_predict = ln.predict(x_test.values)
print('Mean Sqaured Error',mean_squared_error(y_test,y_predict))
print('Mean Absolute Error',mean_absolute_error(y_test,y_predict))
residuals = y_test - y_predict