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
path = r'C:\Users\pmspr\Documents\HS\MS\Sem 2\EECS 738\Lab\1\Work\Code\Data'

#load data using load class and print describe of data
from projectFunctions import loadData
filename = "MotorInsuranceFraudClaimABTFull.csv"
col_names = ['Id', 'InsuranceType', 'Income', 'status', 'claimants', 'Injury', 
             'HospitalStay', 'ClaimAmt', 'TotalClaimed', 'NumberClaims', 
             'NumSoftTissues', 'PerSoftTissue','AmtRecieved','FraudFlag']
data = loadData(path,filename)

# Success - Display the first record
if data is not None:
    data.columns = col_names
    display(data.head(n=1))
    str(data)

#explore the data
from projectFunctions import exploreData
exploreData(data)
#
drop_col = ['Id','InsuranceType','status','NumSoftTissues', 'PerSoftTissue','AmtRecieved','FraudFlag']
features_raw = data.drop(drop_col, axis = 1)
target_raw = data['FraudFlag']
#if features_raw is not None:
#    display(features_raw.head(n=1))
    
#transform data
from projectFunctions import transformData
features,target = transformData(features_raw,target_raw)
#features['NumSoftTissues'] = np.nan_to_num(features['NumSoftTissues'])
 
#shuffle and split the data to create train and test datasets
from projectFunctions import splitData
X_train, X_test, y_train, y_test = splitData(features,target,0.3)
 
from projectFunctions import decTree,drawTree
sample_size = len(X_train)
feature_cols = features.columns

#Usin gini and depth = 3 
results,learner = decTree(sample_size, X_train, y_train, X_test, y_test, 'gini', 20)
drawTree(learner,feature_cols, 'ifraud_gini.png')
print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"

#Usin entropy and depth = 3 
results,learner = decTree(sample_size, X_train, y_train, X_test, y_test, 'entropy', 20)
drawTree(learner,feature_cols, 'ifraud_etropy.png')
print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"
g_train = [0.69714,0.72857,0.75714,0.78571,0.80571,0.83143,0.85429,0.90000,0.91143,0.92286,0.98286,1.00000]
g_test = [0.68000,0.73333,0.69333,0.67333,0.66000,0.65333,0.61333,0.58667,0.54667,0.53333,0.56000,0.52667]
e_train = [0.69714,0.72857,0.75429,0.76286,0.79143,0.81714,0.84286,0.86857,0.90286,0.92857,0.96000,0.99143]
e_test = [0.68000,0.73333,0.70000,0.66667,0.65333,0.62667,0.64000,0.59333,0.55333,0.58667,0.56667,0.56000]
depth = [1,2,3,4,5,6,7,8,9,10,15,20]
plt.plot(depth,g_train,'b',label="Gini_train")
plt.plot(depth,g_test,'r',label="Gini_test")
plt.plot(depth,e_train,'g',label="Entropy_train")
plt.plot(depth,e_test,'y',label="Entropy_test")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot")
plt.legend(loc="upper left")
plt.show()