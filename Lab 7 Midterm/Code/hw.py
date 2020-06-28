# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:46:55 2020

@author: pmspr
"""
import os
import pandas as pd
import numpy  as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors,KNeighborsRegressor

col = ['Ex','Sm','Ob','Fa']
data = pd.DataFrame(columns=col)
#data['Ex'] = ['daily','weekly','daily','rarely','rarely']
data['Ex'] = [3,2,3,1,1]
data['Sm'] = [False,True,False,True,True]
data['Ob'] = [False,False,False,True,True]
#data['Fa'] = ['yes','yes','no','yes','no']
data['Fa'] = [2,2,1,2,1]

target = [0,1,0,1,1]
clf = RandomForestClassifier(criterion='entropy',max_depth=2, random_state=3,bootstrap=True,max_features='sqrt')
clf.fit(data,target)
print(clf.feature_importances_)
print(clf.predict([[1,False,True,2]]))

#tree1
col = ['ex','fa']
dt1 = pd.DataFrame(columns=col)
dt1['ex'] = [3,2,2,1,1]
dt1['fa'] = [2,2,2,1,1]
target    = [0,1,1,1,1]
clf1 = DecisionTreeClassifier(criterion='entropy',max_depth=2, random_state=3)
clf1.fit(dt1,target)
print(clf1.predict([[1,2]]))

#tree2
col = ['sm','ob']
dt2 = pd.DataFrame(columns=col)
dt2['sm'] = [False,True,True,True,True]
dt2['ob'] = [False,False,False,True,True]
target    = [0,1,1,1,1]
clf2 = DecisionTreeClassifier(criterion='entropy',max_depth=2, random_state=3)
clf2.fit(dt2,target)
print(clf2.predict([[False,True]]))

#tree3
col = ['ob','fa']
dt3 = pd.DataFrame(columns=col)
dt3['ob'] = [False,False,False,True,True]
dt3['fa'] = [2,2,2,2,1]
target    = [0,0,1,1,1]
clf3 = DecisionTreeClassifier(criterion='entropy',max_depth=2, random_state=3)
clf3.fit(dt3,target)
print(clf3.predict([[True,2]]))

#Question4
drop_col = ['COUNTRY','CPI']
path = r'C:\Users\pmspr\Documents\HS\MS\Sem 2\EECS 738\Lab\Midterm\Code\Data'
filename = "Cpi.csv"
data = pd.read_csv(os.path.join(path,filename))
#display(data)
X_train = data.drop(drop_col, axis = 1)
y_train = data['CPI']
#
X_test = pd.DataFrame(columns=X_train.columns)
X_test['LIFE']      = [67.62]#[0.9617]#[0.3940]#[78.51]#
X_test['INCOME']    = [31.68]#[0.2242]#[0.0445]#[29.85]#
X_test['MORTALITY'] = [10.00]#[0.0312]#[0.8965]#[6.30]#
X_test['SPEND']     = [3.87]#[0.1547]#[0.6507]#[4.72]#
X_test['SCHOOL']    = [12.90]#[0.8623]#[0.0000]#[13.70]#
#
reg = KNeighborsRegressor(n_neighbors=16,weights='distance',p=2,metric='minkowski')
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
y_nn = reg.kneighbors(X_test, return_distance=False)
print y_pred
print y_nn