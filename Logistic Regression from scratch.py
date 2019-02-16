import numpy as np
from sklearn import datasets
import pandas as pd
class LogisticRegression:
    def __init__(self,eta=0.1,c=1):
        self.eta=eta
        self.n_iter=10
        self.lambdal=1/c

    def fit(self,X,Y):
        self.weights=np.zeros(1+X.shape[1])
        self.cost=[]

        for _ in range(self.n_iter):
         for i in range(self.n_iter):
                Z=self.WTZ(X)
                phiZ=self.sigmoid(Z)
                errors=Y-phiZ
                self.weights[1:]+=self.eta*X.T.dot(errors)
                self.weights[0]+=self.eta*errors.sum()
                cost=((-Y * np.log(phiZ) - (1 - Y) * np.log(1 - phiZ)).mean())/(self.lambdal/2.0)*(np.square(self.weights))
                self.cost.append(cost)
        return self

    def sigmoid(self,Z):
       return 1/(1+np.exp(-Z))
    def WTZ(self,X):
        return np.dot(X,self.weights[1:])+self.weights[0]

    def predict(self,X):
        return np.where(self.sigmoid(self.WTZ(X))>=0.5,1,-1)
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=df.iloc[0:100,[0,2]].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_std_train=sc.transform(X_train)
X_std_test=sc.transform(X_test)


ll=LogisticRegression(c=1000)
ll.fit(X_std_train,Y_train)
y_pred=ll.predict(X_std_test)
print('Misclassified samples: %d' % (Y_test != y_pred).sum())
print(y_pred)
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(Y_test, y_pred))














