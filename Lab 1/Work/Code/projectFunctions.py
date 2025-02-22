# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:20:49 2017

@author: Madhu
"""
import os
import pandas as pd
import numpy  as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.preprocessing import KBinsDiscretizer
import time
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder

#Function to load the data
def loadData(path,filename):
    try:
             files = os.listdir(path)
             for f in files:
                 if f == filename:
                     data = pd.read_csv(os.path.join(path,f))
                     return data
            
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message

#Function to explore the data
def exploreData(data):
    try:
           #Total number of records                                  
           rows = data.shape[0]
           cols = data.shape[1]          
          
           # Print the results
           print "-----------------------------------------------------------------------"
           print "Total number of records: {}".format(rows)
           print "Total number of features: {}".format(cols)
           print "-----------------------------------------------------------------------"
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message
           
#split the data in to train and test data
def splitData(features,target,testsize):
    try:
        # Split the 'features' and 'income' data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target, 
                                                    test_size = testsize, 
                                                    random_state = 1)

        # Show the results of the split
        print "Training set has {} samples.".format(X_train.shape[0])
        print "Testing set has {} samples.".format(X_test.shape[0])
        print "-----------------------------------------------------------------------"
        return X_train, X_test, y_train, y_test
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message

def transformData(features,target):
    try:
        # Discretize continuous feature 
#        dv = Discretizer(features['AmtRecieved'])
#        features['AmtRecieved'] = dv
        
        skewed = ['Income','TotalClaimed']
        features_log_transformed = pd.DataFrame(data = features)
        features_log_transformed[skewed] = features[skewed].apply(lambda x: np.log(x + 1))
        
        scaler = MinMaxScaler() # default=(0, 1)
        numerical = ['ClaimAmt']
        features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
        features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
        
#        #replace zeros (outliers) with mean
#        li= list(features_log_minmax_transform['AmtRecieved'])
#        meanvalue = sum(li) / float(len(li))
#        tlist = list(features_log_minmax_transform['AmtRecieved'][features_log_minmax_transform['AmtRecieved']==0].index)
#        features_log_minmax_transform.loc[tlist,['AmtRecieved']] = meanvalue
#        #dv = Discretizer(features_log_minmax_transform['AmtRecieved'])
#        features_log_minmax_transform['AmtRecieved'] = dv
        
        # TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
        #features_final = pd.get_dummies(features_log_minmax_transform)
        enc = LabelEncoder()
        features_log_minmax_transform['Injury'] = enc.fit_transform(features_log_minmax_transform['Injury'])
        features_log_minmax_transform['HospitalStay'] = enc.fit_transform(features_log_minmax_transform['HospitalStay'])
        features_final = features_log_minmax_transform
        # Print the number of features after one-hot encoding
        #encoded = list(features_final.columns)
        #print "{} total features after one-hot encoding.".format(len(encoded))

        # Uncomment the following line to see the encoded feature names
        #print encoded
        return features_final, target
        
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message
           
def decTree(sample_size, X_train, y_train, X_test, y_test, method, depth):
    try:
        #Decision tree classifier
        learner = DecisionTreeClassifier(criterion=method, max_depth=depth, random_state=1)
        results = {}
         
        start_time = time.clock()
        clf_fit_train = learner.fit(X_train[:sample_size], y_train[:sample_size])
        end_time = time.clock()
        results['train_time'] = end_time - start_time
               
        start_time = time.clock()
        clf_predict_test = clf_fit_train.predict(X_test)
        clf_predict_train = clf_fit_train.predict(X_train[:sample_size])
        end_time = time.clock()
        results['pred_time'] = end_time - start_time  
         
        results['acc_train'] = accuracy_score(y_train[:sample_size], clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
        results['f_train']   = fbeta_score(y_train[:sample_size], clf_predict_train, average='binary', beta=0.5)
        results['f_test']    = fbeta_score(y_test, clf_predict_test, average='binary', beta=0.5)
            
        return results,learner      
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message
        
def drawTree(clf,feature_cols,fname):
    try:
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True,feature_names = feature_cols,class_names=['0','1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(fname)
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message
        
def Discretizer(col):
    try:
        # transform the dataset with KBinsDiscretizer
        dat = np.array(col.tolist()).reshape(-1,1)
        enc = KBinsDiscretizer(encode='ordinal' ,strategy='uniform')
        enc.fit(dat)
        X_binned = enc.transform(dat)
        return X_binned
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message