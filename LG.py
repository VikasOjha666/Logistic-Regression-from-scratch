import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class LogisticRegression:
 def __init__(self,eta=0.01,epochs=10):
     self.eta=eta
     self.epochs=epochs
 def sigmoid(self,x):
  return 1 / (1 + np.exp(-x))


 def fit(self,X,Y):
     self.W=np.zeros(1+X.shape[1])
     self.cost=[]
     for _ in range(self.epochs):

         pred=self.predict(X)
         errors=Y-pred
         self.W[1:]+=self.eta*X.T.dot(errors)
         self.W[0]=self.eta*errors.sum()
         cost=cost=((-Y * np.log(pred) - (1 - Y) * np.log(1 - pred)).mean())/(np.square(self.W))
         self.cost.append(cost)
 def predict(self,X):
     prod=np.dot(X,self.W[1:])+self.W[0]
     prod=self.sigmoid(prod)
     return np.where(prod>=0.5,1,-1)



data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y = data.iloc[0:100, 4].values
y=np.where(y=='Iris-setosa',-1,1)
X = data.iloc[0:100, [0, 2]].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_std_train=sc.transform(X_train)
X_std_test=sc.transform(X_test)

X_combined_std = np.vstack((X_std_train, X_std_test))
y_combined = np.hstack((Y_train, Y_test))

ll=LogisticRegression()
ll.fit(X_std_train,Y_train)
y_pred=ll.predict(X_std_test)
print('Misclassified samples: %d' % (Y_test != y_pred).sum())
print(y_pred)


def plot_decision_regions(X, y, classifier, resolution=0.02):
#This functions has been taken from python Machine Learning book by Sebestian Rashchka.
# setup marker generator and color map
 markers = ('s', 'x', 'o', '^', 'v')
 colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
 cmap = ListedColormap(colors[:len(np.unique(y))])
 # plot the decision surface
 x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
 x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
 np.arange(x2_min, x2_max, resolution))
 Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
 Z = Z.reshape(xx1.shape)
 plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
 plt.xlim(xx1.min(), xx1.max())
 plt.ylim(xx2.min(), xx2.max())

 # plot class samples
 for idx, cl in enumerate(np.unique(y)):
  plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)




plot_decision_regions(X_combined_std, y_combined, classifier=ll)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
