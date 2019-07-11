import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

dfx=pd.read_csv('Linear_X_Train.csv')
dfy=pd.read_csv('Linear_Y_Train.csv')

X=dfx.values
Y=dfy.values
print(X.shape)
print(Y.shape)

Y=Y.reshape((-1,))
lr=LinearRegression()

lr.score(X,Y)
dXtest=pd.read_csv('Linear_X_Test.csv')
Xtest=dXtest.values
n=Xtest.shape[0]
Xtest=np.array(Xtest)
Xtest=Xtest.reshape((-1,1))

Y=[]
import csv
with open('linearanswerspredicted.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    for i in range(n):
        y=lr.predict([Xtest[i]])
        writer.writerows([y])
        Y.append(y)
print(Y)
csvFile.close()