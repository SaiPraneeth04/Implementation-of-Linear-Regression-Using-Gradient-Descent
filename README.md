# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard python libraries for Gradient design.
2. Introduce the variables needed to execute the function.
3. Use function for the representation of the graph.
4. Using for loop apply the concept using the formulae.
5. Execute the program and plot the graph.
6. Predict and execute the values for the given conditions.

## Program:
```python
Program to implement the linear regression using gradient descent.
Developed by: Sai Praneeth K
RegisterNumber:  212222230067
```
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)
data


plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("profit($10,000)")
plt.title("Profit prediction")


def computeCost(X,y,theta):
    m=len(y)
    h=X.dot(theta)
    square_err=(h-y)**2
    return 1/(2*m) * np.sum(square_err)


    data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
X.shape

y.shape

theta.shape

computeCost(X,y,theta)


def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = X.dot(theta)
    error = np.dot(X.transpose(),(predictions -y))
    descent=alpha *1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta, J_history


theta,J_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")


plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")


def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]


predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![output](/Screenshot%20(147).png)

![output](/Screenshot%20(148).png)

![output](/Screenshot%20(149).png)

![output](/Screenshot%20(150).png)

![output](/Screenshot%20(151).png)

![output](/Screenshot%20(152).png)

![output](/Screenshot%20(153).png)

![output](/Screenshot%20(154).png)

![output](/Screenshot%20(155).png)

![output](/Screenshot%20(156).png)

![output](/Screenshot%20(157).png)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
