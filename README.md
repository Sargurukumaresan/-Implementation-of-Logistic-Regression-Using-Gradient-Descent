# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the data file and import numpy, matplotlib and scipy.

2.Visulaize the data and define the sigmoid function, cost function and gradient descent.

3.Plot the decision boundary.

4.Calculate the y-prediction.

## Program:
```python
'''
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SARGURU K
RegisterNumber: 212222230134
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset['gender'] = dataset['gender'].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta ,X ,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iters):
    m=len(y)
    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -=alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iters=1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:
### dataset
![image](https://github.com/Madhavareddy09/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742470/8df2bfc9-121c-4c30-83b9-0e3d880c183c)

### datatypes
![image](https://github.com/Madhavareddy09/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742470/4ceed3c3-6976-4fe6-8d88-e9c772c644ea)

### dataset after printing only codes columns
![image](https://github.com/Madhavareddy09/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742470/dde79df5-c8cb-4996-a8de-88d23fb90850)

### Accuracy
![image](https://github.com/Madhavareddy09/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742470/633706d9-e413-4759-bff3-eca1946864a8)

### Array values of Y prediction
![image](https://github.com/Madhavareddy09/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742470/9f957773-e789-4f87-88d7-8ee7d8e54a80)

### Array values of Y
![image](https://github.com/Madhavareddy09/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742470/56420e72-baf9-46c4-9229-f3244b238523)

### predicting with different values
![image](https://github.com/Madhavareddy09/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742470/6f6447d1-5e0c-4267-883f-734d60ac19d9)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

