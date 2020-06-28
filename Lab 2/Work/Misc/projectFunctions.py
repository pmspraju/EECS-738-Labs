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
from sklearn.metrics import accuracy_score,r2_score
from sklearn.metrics import fbeta_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import KBinsDiscretizer
import time
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors,KNeighborsRegressor

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

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
           
           #print histograms of columns
           drawCorr(data)
           
           #draw correlation
           plt.figure(figsize=(13,13))
           sns.heatmap(data.corr(),
           vmin=-1,
           cmap='coolwarm',
           annot=True);
           
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
                
        scaler = MinMaxScaler() # default=(0, 1)
        numerical = ['FFMC','DMC','DC','ISI','temp','RH','wind']
        features_log_minmax_transform = pd.DataFrame(data = features)
        features_log_minmax_transform[numerical] = scaler.fit_transform(features[numerical])
                
        # TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
        #features_final = pd.get_dummies(features_log_minmax_transform)
        enc = LabelEncoder()
        features_log_minmax_transform['month'] = enc.fit_transform(features_log_minmax_transform['month'])
        features_log_minmax_transform['day'] = enc.fit_transform(features_log_minmax_transform['day'])
        features_final = features_log_minmax_transform
        # Print the number of features after one-hot encoding
        #encoded = list(features_final.columns)
        #print "{} total features after one-hot encoding.".format(len(encoded))
        
        target_reg = target
        ind = np.where((target>0) & (target<=200))
        target.iloc[ind] = 1    
        ind = np.where((target>200) & (target<=400))
        target.iloc[ind] = 2
        ind = np.where((target>400) & (target<=800))
        target.iloc[ind] = 3
        ind = np.where((target>800))
        target.iloc[ind] = 4
         
        return features_final, target, target_reg
        
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message
           
def decTree(sample_size, X_train, y_train, X_test, y_test, method, depth):
    try:
        #Decision tree classifier
        #learner = DecisionTreeClassifier(criterion=method, max_depth=depth, random_state=1)
        clf = DecisionTreeClassifier(random_state=0)
        params = {'max_depth':[depth],'criterion':[method]}
        #params = {'criterion':['gini','entropy'], 'max_depth' : np.array([6,7,8])}

        scoring_fnc = make_scorer(fbeta_score,average='micro',beta=0.5)
        learner = GridSearchCV(clf,params,scoring=scoring_fnc)
        results = {}
         
        start_time = time.clock()
        grid = learner.fit(X_train,y_train)
        
        end_time = time.clock()
        results['train_time'] = end_time - start_time
        clf_fit_train = grid.best_estimator_
        start_time = time.clock()
        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
        end_time = time.clock()
        results['pred_time'] = end_time - start_time  
         
        results['acc_train'] = accuracy_score(y_train[:sample_size], clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
        results['f_train']   = fbeta_score(y_train[:sample_size], clf_predict_train, average='micro', beta=1)
        results['f_test']    = fbeta_score(y_test, clf_predict_test, average='micro', beta=0.5)
        
        return results,clf_fit_train      
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

def decTreeReg(X_train, y_train, X_test, y_test, method, depth):
    try:
        learner = DecisionTreeRegressor(max_depth=depth)
        clf_fit_train = learner.fit(X_train, y_train)
        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
        results = {}
        results['acc_train'] = r2_score(y_train, clf_predict_train)
        results['acc_test']  = r2_score(y_test, clf_predict_test)
        
        return results,learner
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message    
    

def kneighbors(X_train, y_train, X_test, y_test):
    try:
        clf = KNeighborsClassifier(n_neighbors=7,metric='euclidean')
        clf.fit(X_train,y_train)
        clf_predict_train = clf.predict(X_train)
        clf_predict_test = clf.predict(X_test)
        results = {}
        results['acc_train'] = accuracy_score(y_train, clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
        results['f_train']   = fbeta_score(y_train, clf_predict_train, average='micro', beta=1)
        results['f_test']    = fbeta_score(y_test, clf_predict_test, average='micro', beta=0.5)
        return results

    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message

def kneighbhorsReg(X_train, y_train, X_test, y_test):
    try:
        clf = KNeighborsRegressor(n_neighbors=1)
        clf.fit(X_train,y_train)
        clf_predict_train = clf.predict(X_train)
        clf_predict_test = clf.predict(X_test)
        results = {}
        results['acc_train'] = r2_score(y_train, clf_predict_train)
        results['acc_test']  = r2_score(y_test, clf_predict_test)
        
        return results,clf
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message    

def drawCorr(df):
    try:
        #df.hist(column='X',bins=70)   
        fig = plt.figure()
        ax1 = plt.subplot(3,4,1)
        df['X'].value_counts().plot(kind='bar')
        ax1.set_title('X')
        
        ax2 = plt.subplot(3,4,2)
        df['Y'].value_counts().plot(kind='bar')
        ax2.set_title('Y')
        
        ax3 = plt.subplot(3,4,3)
        df['month'].value_counts().plot(kind='bar')
        ax3.set_title('month')
        
        ax4 = plt.subplot(3,4,4)
        df['day'].value_counts().plot(kind='bar')
        ax4.set_title('day')
        
        ax5 = plt.subplot(3,4,5)
        df['FFMC'].plot()
        ax5.set_title('FFMC')
        
        ax6 = plt.subplot(3,4,6)
        df['DMC'].plot()
        ax6.set_title('DMC')
        
        ax7 = plt.subplot(3,4,7)
        df['DC'].plot()
        ax7.set_title('DC')
        
        ax8 = plt.subplot(3,4,8)
        df['ISI'].plot()
        ax8.set_title('ISI')
        
        ax9 = plt.subplot(3,4,9)
        df['temp'].plot()
        ax9.set_title('temp')
        
        ax10 = plt.subplot(3,4,10)
        df['RH'].plot()
        ax10.set_title('RH')
        
        ax11 = plt.subplot(3,4,11)
        df['wind'].plot()
        ax11.set_title('wind')
        
        ax12 = plt.subplot(3,4,12)
        df['rain'].plot()
        ax12.set_title('rain')
        
        plt.show()
             
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message        