# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sai Praneeth K
RegisterNumber: 212222230067
*/
```
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  x=np.c_[np.ones(len(x1)),x1]
  theta=np.zeros(x.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(x).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
  return theta

data=pd.read_csv('50_Startups.csv',header=None)
data.head()

x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_Scaled=scaler.fit_transform(x1)
y1_Scaled=scaler.fit_transform(y)

theta=linear_regression(x1_Scaled, y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```
## Output:
## data table:
![3 1](https://github.com/SaiPraneeth04/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119390353/cfe25db4-b090-4fd9-8532-700a9e2cdb73)

## predicted value:
![3 2](https://github.com/SaiPraneeth04/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119390353/5af15b35-3d0b-463f-85df-4ef472e88b3e)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
