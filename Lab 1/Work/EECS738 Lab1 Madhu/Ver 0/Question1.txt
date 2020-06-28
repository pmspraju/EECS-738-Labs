# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn

print "-----------------------------------------------------------------------"
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#get the working directory and filename
path = r'C:\Users\pmspr\Documents\HS\MS\Sem 2\EECS 738\Lab\1\Work\Code\Data'

#load data using load class and print describe of data
from projectFunctions import loadData
filename = "diabetes.csv"
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
data = loadData(path,filename)

# Success - Display the first record
if data is not None:
    data.columns = col_names
    #display(data.head(n=1))

#explore the data
from projectFunctions import exploreData
exploreData(data)

drop_col = ['skin','label']
features = data.drop(drop_col, axis = 1)
target = data['label']
#if features is not None:
    #display(features.head(n=1))
#
#shuffle and split the data to create train and test datasets
from projectFunctions import splitData
X_train, X_test, y_train, y_test = splitData(features,target,0.3)

from projectFunctions import decTree,drawTree
sample_size = len(X_train)
feature_cols = features.columns
 
results,learner = decTree(sample_size, X_train, y_train, X_test, y_test, 'gini', 3)
drawTree(learner,feature_cols, 'diabetes.png')
print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"