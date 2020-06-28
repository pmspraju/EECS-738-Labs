# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:29:23 2020

@author: pmspr
"""
# Import libraries necessary for this project
import sklearn
import seaborn as sns; sns.set()
print "-----------------------------------------------------------------------"
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))
 
#get the working directory and filename
path = r'C:\Users\pmspr\Documents\HS\MS\Sem 2\EECS 738\Lab\Midterm\Code\Data'

#load data using load class and print describe of data
from featureEng import loadData
filename = "ACMETelephoneABT.csv"

data = loadData(path,filename)
 
##explore the data
from featureEng import exploreData
features_raw,target_raw = exploreData(data)
 
drop_col = ['customer','occupation']
#drop_col = ['customer','occupation','marriageStatus','children',
#            'smartPhone','creditRating','homeOwner','creditCard']
features_raw = features_raw.drop(drop_col, axis = 1)
  
#transform data
from projectFunctions import transformData
features,target = transformData(features_raw,target_raw)
 
# Success - Display the first record
#if data is not None:
#    display(data.head(n=1))
#    print data.describe(include='all')
#if features_raw is not None:
#    display(features_raw.head(n=1))    
#
## 
#shuffle and split the data to create train and test datasets
from projectFunctions import splitData
X_train, X_test, y_train, y_test = splitData(features,target,0.3)
 
from projectFunctions import decTree,randomForest,kneighbors,drawTree,svmClass,pcaComp,neunet
sample_size = len(X_train)

#Apply PCA to reduce the dimensions
#X_train, X_test = pcaComp(X_train, X_test,30)
#X_train.to_csv('pca_dim.csv')

results,imp_features = randomForest(X_train, y_train, X_test, y_test)
print "Accuracy for Random forest Classifier - Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"

#Use only imporatant features from random forest
#X_train = X_train[imp_features]
    #X_test  = X_test[imp_features]

#Usin gini and depth = 3 
results,learner = decTree(sample_size, X_train, y_train, X_test, y_test, 'entropy', 4)
feature_cols = X_train.columns
feature_cols = [x.encode('utf-8') for x in feature_cols]
 
drawTree(learner,feature_cols, 'churn.png')
print "Accuracy for Decision tree Classifier - Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"

#kneighbors classifier
resultsK = kneighbors(X_train, y_train, X_test, y_test)
print "Accuracy for K-Neighbors Classifier-Training, Test sets: %.5f, %.5f" %(resultsK['acc_train'], resultsK['acc_test'])     
print "-----------------------------------------------------------------------"

#SVM classifier
resultsS = svmClass(X_train, y_train, X_test, y_test)
print "Accuracy for SVM Classifier-Training, Test sets: %.5f, %.5f" %(resultsS['acc_train'], resultsS['acc_test'])     
print "-----------------------------------------------------------------------"

#neural net classifier with back propogation
resultsN = neunet(X_train, y_train, X_test, y_test)
print "Accuracy for Neural Net Classifier-Training, Test sets: %.5f, %.5f" %(resultsN['acc_train'], resultsN['acc_test'])     
print "-----------------------------------------------------------------------"