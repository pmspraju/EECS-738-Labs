# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn
import numpy  as np
from matplotlib import pyplot as plt
print "-----------------------------------------------------------------------"
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#get the working directory and filename
path = r'C:\Users\pmspr\Documents\HS\MS\Sem 2\EECS 738\Lab\2\Work\Code\Data'

#load data using load class and print describe of data
from projectFunctions import loadData
filename = "forestfires.csv"

data = loadData(path,filename)

##explore the data
from projectFunctions import exploreData
exploreData(data)

# Success - Display the first record
if data is not None:
    display(data.head(n=1))
    print data.describe(include='all')

drop_col = ['X','Y','rain','area']
features_raw = data.drop(drop_col, axis = 1)
target_raw = data['area']
if features_raw is not None:
    display(features_raw.head(n=1))    

#transform data
from projectFunctions import transformData
features,target,target_reg = transformData(features_raw,target_raw)

# 
##shuffle and split the data to create train and test datasets
from projectFunctions import splitData
X_train, X_test, y_train, y_test = splitData(features,target,0.3)
Xr_train, Xr_test, yr_train, yr_test = splitData(features,target_reg,0.3)
# 
from projectFunctions import decTree,drawTree,kneighbors,decTreeReg,kneighbhorsReg
sample_size = len(X_train)
feature_cols = features.columns
 
#Usin gini and depth = 3 
results,learner = decTree(sample_size, X_train, y_train, X_test, y_test, 'entropy', 4)
drawTree(learner,feature_cols, 'fire_dt.png')
print "Accuracy for Decision tree Classifier - Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"

#decision tree regression
results_dreg, learner_dreg = decTreeReg(Xr_train, yr_train, Xr_test, yr_test, 'mse', 4)
print "R2 score for Decision tree regression -Training, Test sets: %.5f, %.5f" %(results_dreg['acc_train'], results_dreg['acc_test'])     
print "-----------------------------------------------------------------------"

#kneighbors classifier
resultsK = kneighbors(X_train, y_train, X_test, y_test)
print "Accuracy for K-Neighbors Classifier-Training, Test sets: %.5f, %.5f" %(resultsK['acc_train'], resultsK['acc_test'])     
print "-----------------------------------------------------------------------"

#kneighbors tree regression
results_kreg, learner_kreg = kneighbhorsReg(Xr_train, yr_train, Xr_test, yr_test)
print "R2 score for K Neighbhors regression -Training, Test sets: %.5f, %.5f" %(results_kreg['acc_train'], results_kreg['acc_test'])     
print "-----------------------------------------------------------------------"