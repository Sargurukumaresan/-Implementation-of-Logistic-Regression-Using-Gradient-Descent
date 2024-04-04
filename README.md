# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SARGURU K
RegisterNumber: 212222230134

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
*/
```

## Output:
Array Value of x:


![image](https://user-images.githubusercontent.com/119389139/233685454-0adb4b94-f3fd-4e97-b8aa-6fcefb043d74.png)


## Array Value of y:

![image](https://user-images.githubusercontent.com/119389139/233685521-644c69ed-e33f-4c87-9328-fc0bef73714c.png)

## Exam 1 - score graph:




![image](https://user-images.githubusercontent.com/119389139/233688106-0246eac6-0a24-4b94-b162-1c08c4777107.png)


## Sigmoid function graph:


![image](https://user-images.githubusercontent.com/119389139/233685832-e45b2e1b-aaf0-4828-bb22-eb845e32030b.png)


## X_train_grad value:


![image](https://user-images.githubusercontent.com/119389139/233686027-0877fcff-43e1-458b-9c76-775fd2c09e98.png)


## Y_train_grad value:


![image](https://user-images.githubusercontent.com/119389139/233686071-d4e01b9b-5bf1-4b47-b546-9f82e30c6213.png)


## Print res.x:


![image](https://user-images.githubusercontent.com/119389139/233686143-e28eb2d2-b7b7-42ff-9ce9-6009bd8ebfc5.png)


## Decision boundary - graph for exam score:


![image](https://user-images.githubusercontent.com/119389139/233686223-36c51815-9370-4076-885a-85c553eee393.png)


## Proability value:

![image](https://user-images.githubusercontent.com/119389139/233686314-7faa0663-8a83-49a7-9194-3941b2733b51.png)


## Prediction value of mean:

![image](https://user-images.githubusercontent.com/119389139/233686372-15b81f06-74a3-4c98-b2c2-278b326ee179.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
