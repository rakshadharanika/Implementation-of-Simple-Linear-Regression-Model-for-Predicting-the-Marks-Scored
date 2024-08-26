# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values.  
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by  : RAKSHA DHARANIKA V
RegisterNumber: 212223230167

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df
df.head()
df.tail()
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='pink')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='gold')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### Dataset
![image](https://github.com/user-attachments/assets/bd11aa0b-8ceb-43e6-95c6-110b63135098)
### Head values
![image](https://github.com/user-attachments/assets/45e23d3f-9e16-483d-88f2-c40920cdbd14)

### Tail values
![image](https://github.com/user-attachments/assets/6f60969c-bd34-4063-b1fe-38a4d44ff38f)

### X and Y values
![image](https://github.com/user-attachments/assets/ce05a213-e361-407c-86a3-4a35628bdbce)




![image](https://github.com/user-attachments/assets/a05f4c16-655a-4813-a312-02b2c69f8f75)


### Predication values of X and Y


![image](https://github.com/user-attachments/assets/4bb21001-4b3e-4631-b320-42836d1fad63)




![image](https://github.com/user-attachments/assets/8c0a09e0-35f0-49ac-b150-ac86979506dc)


### MSE,MAE and RMSE:
![image](https://github.com/user-attachments/assets/0a94814b-84e6-4a6c-b4d0-6fa50510db01)

### Training Set:
![image](https://github.com/user-attachments/assets/634abdb2-141f-4689-96e4-314f7b0263ee)

### Testing Set:
![image](https://github.com/user-attachments/assets/98864a74-952f-4892-836f-308400ac8d22)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
