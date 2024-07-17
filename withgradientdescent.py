from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('housing.csv')
x = data[['median_income','households']].values
y = data['median_house_value'].values
ones = np.ones((x.shape[0],1))
x = np.hstack((ones,x))
print(x)
fig = plt.figure()
ax = plt.axes(projection ='3d')
z = y
y_ = x[:,0]
x = x[:,1]

ax.plot3D(x,y,z)
ax.set_title('Sample')
plt.show()