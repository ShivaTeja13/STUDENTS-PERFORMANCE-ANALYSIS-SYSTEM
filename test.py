import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('Extension_dataset.csv')
dataset.fillna(0, inplace = True)
labels = ['excellent', 'medium', 'poor']
cols = ['becgrade','Placed','becstatus']
le = LabelEncoder()
dataset[cols[0]] = pd.Series(le.fit_transform(dataset[cols[0]].astype(str)))
dataset[cols[1]] = pd.Series(le.fit_transform(dataset[cols[1]].astype(str)))
dataset[cols[2]] = pd.Series(le.fit_transform(dataset[cols[2]].astype(str)))

cols = dataset.shape[1]-1
dataset = dataset.values
X = dataset[:,0:cols]
Y = dataset[:,cols]
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(X)
print(Y)

cls = DecisionTreeClassifier(criterion = "gini", max_depth = 200,splitter="best",class_weight="balanced",max_leaf_nodes=100)
cls.fit(X,Y)
predict = cls.predict(X_test) 
a = accuracy_score(y_test,predict)*100
print(a)
print("\n")

for i in range(0,5):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    cls = DecisionTreeClassifier(max_depth = 200,splitter="best",class_weight="balanced",max_leaf_nodes=100)
    cls.fit(X,Y)
    predict = cls.predict(X_test) 
    a = accuracy_score(y_test,predict)*100
    print(a)
    
