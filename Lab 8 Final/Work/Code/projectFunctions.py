# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:20:49 2017

@author: Madhu
"""
import pandas as pd
import numpy  as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder
import seaborn as sns; sns.set()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

def transformData(features,target):
    try:
         
        ind_true = target[target == True].index
        ind_false = target[target == False].index
        
        target.loc[target == False] = 0
        target.loc[target == True] = 1
 
        ##CATEGORICAL FEATURES********
        #change different categorical names to relevant categories
        features.loc[features.regionType == 's','regionType'] = 'suburban'        
        features.loc[features.regionType == 't','regionType'] = 'town'
        features.loc[features.regionType == 'r','regionType'] = 'rural'
        
        features.loc[features.creditCard == 't','creditCard'] = 'true'
        features.loc[features.creditCard == 'yes','creditCard'] = 'true'
        features.loc[features.creditCard == 'f','creditCard'] = 'false'
        features.loc[features.creditCard == 'no','creditCard'] = 'false'
        
        #replace missing values with most freguent values groupedby churn value
        mind = features['regionType'][features['regionType'] != features['regionType']].index
        ind_f = (set(mind) & set(ind_false))
        features['regionType'][ind_f] = 'town'
         
        ind_t = (set(mind) & set(ind_true))
        features['regionType'][ind_t] = 'suburban'
         
        #one-hot encoding for categorical values
        features_encode = pd.DataFrame(data = features)
        enc = LabelEncoder()

        features_encode['regionType'] = enc.fit_transform(features_encode['regionType'])
        ###
        categorical = ['regionType','marriageStatus','children','smartPhone','creditRating','homeOwner','creditCard']
        en_df = onehotencode(categorical,features_encode)
        features_encode = features_encode.merge(en_df, left_on = 'regionType', right_index = True, how = 'left')
        ###
        li = ['marriageStatus','children','smartPhone','creditRating','homeOwner','creditCard']
        features_encode = features_encode.drop(li,axis=1)
            
        ###NUMERICAL FEATURES******** 
        #correct negative values
        features_encode.loc[features_encode.handsetAge <0, 'handsetAge'] = 0
        #Apply log transformation for skewed features with outliers
        features_log_transformed = pd.DataFrame(data = features_encode)
        features_log_transformed['callMinutesChangePct'] = features_encode['callMinutesChangePct'].apply(lambda x: np.log(x + 50.4))
        features_log_transformed['billAmountChangePct'] = features_encode['billAmountChangePct'].apply(lambda x: np.log(x + 7.61))
        features_log_transformed['peakOffPeakRatioChangePct'] = features_encode['peakOffPeakRatioChangePct'].apply(lambda x: np.log(x + 41.33))

        skewed = ['handsetAge','currentHandsetPrice','avgrecurringCharge','avgOverBundleMins',
                  'avgRoamCalls','avgReceivedMins','avgOutCalls','avgInCalls',
                  'peakOffPeakRatio','avgDroppedCalls','lifeTime','lastMonthCustomerCareCalls',
                  'numRetentionCalls','numRetentionOffersAccepted','newFrequentNumbers']
        features_log_transformed[skewed] = features_encode[skewed].apply(lambda x: np.log(x + 0.1))
        
        scaler = MinMaxScaler() # default=(0, 1)
        numerical = features_log_transformed.columns
        features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
        features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
         
        #drop columns if any
        final_dropcol = ['lastMonthCustomerCareCalls','numRetentionCalls','numRetentionOffersAccepted','newFrequentNumbers']        
        features_final = features_log_minmax_transform.drop(final_dropcol, axis = 1)
        
#        # Print the number of features after one-hot encoding
#        #encoded = list(features_final.columns)
#        #print "{} total features after one-hot encoding.".format(len(encoded))       
#        from featureEng import printStat
#        printStat(features_final)
#        from featureEng import missingValues
#        missingValues(features_encode)   
        
        features_final.to_csv('transformed.csv')
        return features_final, target
        
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
        print "-----------------------------------------------------------------------"
        print "Training set has {} samples.".format(X_train.shape[0])
        print "Testing set has {} samples.".format(X_test.shape[0])
        print "-----------------------------------------------------------------------"
        return X_train, X_test, y_train, y_test
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message
           
def decTree(sample_size, X_train, y_train, X_test, y_test, method, depth):
    try:
        #Decision tree classifier
        #learner = DecisionTreeClassifier(criterion=method, max_depth=depth, random_state=1)
        clf = DecisionTreeClassifier()
        #params = {'random_state':[4],'max_depth':[depth],'criterion':[method]}
        params = {'criterion':['gini','entropy'], 'max_depth' : np.array([6,7,8])}

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
        results['f_test']    = fbeta_score(y_test, clf_predict_test, average='micro', beta=1.5)
        
#        re = pd.DataFrame(columns=['Actual','Pred'])
#        re['Pred'] = clf_predict_train
#        re['Actual'] = y_train
#        re.to_csv('Results_d.csv')
        
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
                        special_characters=True,feature_names = feature_cols,class_names=['false','true'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(fname)
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message

def randomForest(X_train,y_train,X_test,y_test):
    try:
        clf = RandomForestClassifier(criterion='entropy',max_depth=2, random_state=3,bootstrap=True,max_features='sqrt')
        clf.fit(X_train,y_train)
        clf_predict_train = clf.predict(X_train)
        clf_predict_test = clf.predict(X_test)
        
#        re = pd.DataFrame(columns=['Actual','Pred'])
#        re['Pred'] = clf_predict_train
#        re['Actual'] = y_train
#        re.to_csv('Results_r.csv')
        
        #Display Important features
        dic = {'feature':X_train.columns, 'Import':clf.feature_importances_}
        f_imp = pd.DataFrame(dic)
        f_imp = f_imp.sort_values(by=['Import'],ascending=False)
        imp_features = f_imp.loc[f_imp.Import > 0, 'feature']
        #print(f_imp)
        
        results = {}          
        results['acc_train'] = accuracy_score(y_train, clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
        results['f_train']   = fbeta_score(y_train, clf_predict_train, average='micro', beta=1)
        results['f_test']    = fbeta_score(y_test, clf_predict_test, average='micro', beta=1.5)
         
        return results,imp_features.tolist()
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message


def kneighbors(X_train, y_train, X_test, y_test):
    try:
        clf = KNeighborsClassifier(n_neighbors=7)
        clf_fit_train=clf.fit(X_train,y_train)
        
        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
        
#        re = pd.DataFrame(columns=['Actual','Pred'])
#        re['Pred'] = clf_predict_train
#        re['Actual'] = y_train
#        re.to_csv('Results_k.csv')
        
        results = {}
        results['acc_train'] = accuracy_score(y_train, clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
        results['f_train']   = fbeta_score(y_train, clf_predict_train, average='micro', beta=1)
        results['f_test']    = fbeta_score(y_test, clf_predict_test, average='micro', beta=1.5)
        return results

    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message

def svmClass(X_train, y_train, X_test, y_test):
    try:
        #print(y_train.unique)
        cw = {1:7,0:1}
        clf = SVC(kernel='poly',degree=2,gamma='auto',random_state=4,class_weight=cw)
        clf_fit_train=clf.fit(X_train,y_train)
        
        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
        
#        re = pd.DataFrame(columns=['Actual','Pred'])
#        re['Pred'] = clf_predict_train
#        re['Actual'] = y_train
#        re.to_csv('Results_s.csv')
        
        results = {}
        results['acc_train'] = accuracy_score(y_train, clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
        results['f_train']   = fbeta_score(y_train, clf_predict_train, average='micro', beta=1)
        results['f_test']    = fbeta_score(y_test, clf_predict_test, average='micro', beta=1.5)
        
        return results
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message    

def onehotencode(colList,df):
    try:
        df.loc[df.children == True,'children'] = 'tru'
        df.loc[df.children == False,'children'] = 'fal'
        df.loc[df.smartPhone == True,'smartPhone'] = 'tru'
        df.loc[df.smartPhone == False,'smartPhone'] = 'fal'
        df.loc[df.homeOwner == True,'homeOwner'] = 'tru'
        df.loc[df.homeOwner == False,'homeOwner'] = 'fal'
        categorical = pd.get_dummies(df[colList])
        categorical_grouped = categorical.groupby('regionType').agg(['sum', 'mean'])
        
        group_var = 'regionType'

        # Need to create new column names
        columns = []
        
        # Iterate through the variables names
        for var in categorical_grouped.columns.levels[0]:
            # Skip the grouping variable
            if var != group_var:
                # Iterate through the stat names
                for stat in ['count', 'count_norm']:
                    # Make a new column name for the variable and stat
                    columns.append('%s_%s' % (var, stat))
        
        #  Rename the columns
        categorical_grouped.columns = columns

#        edf.to_csv('cat.csv')
        
        return categorical_grouped
        
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message    

def pcaComp(X_train,X_test,ncomp):
    try:
        pca = PCA(n_components=ncomp)
        X_train_red = pca.fit_transform(X_train)
        X_test_red = pca.fit_transform(X_test)
        X_train_df = pd.DataFrame(X_train_red,columns=['PCA%i' % i for i in range(ncomp)], index=X_train.index)
        X_test_df = pd.DataFrame(X_test_red,columns=['PCA%i' % i for i in range(ncomp)], index=X_test.index)
        return X_train_df, X_test_df
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message     
        
def neunet(X_train, y_train, X_test, y_test):
    try:
        clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10, 1), random_state=1)
        clf_fit_train = clf.fit(X_train, y_train)

        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
        
#        re = pd.DataFrame(columns=['Actual','Pred'])
#        re['Pred'] = clf_predict_train
#        re['Actual'] = y_train
#        re.to_csv('Results_n.csv')
        
        results = {}
        results['acc_train'] = accuracy_score(y_train, clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
        results['f_train']   = fbeta_score(y_train, clf_predict_train, average='micro', beta=1)
        results['f_test']    = fbeta_score(y_test, clf_predict_test, average='micro', beta=1.5)
        
        return results        
        
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message     