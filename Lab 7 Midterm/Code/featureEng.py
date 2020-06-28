# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:20:49 2017

@author: Madhu
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import pandas.plotting as pp

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

#Function to load the data
def loadData(path,filename):
    try:
             files = os.listdir(path)
             for f in files:
                 if f == filename:
                     data = pd.read_csv(os.path.join(path,f), sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
                     return data
            
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message

#Function to explore the data
def exploreData(data):
    try:
        
           #separate features and target
           drop_col = ['churn']
           features = data.drop(drop_col, axis = 1)
           target = data['churn']
        
           #Total number of records                                  
           rows = data.shape[0]
           cols = data.shape[1]          
          
           # Print the results
           print "-----------------------------------------------------------------------"
           print "Total number of records: {}".format(rows)
           print "Total number of features: {}".format(cols)
           print "-----------------------------------------------------------------------"
           
           #See the balance of target column
           #target_counts= target.value_counts()                      
           #barPlot(target_counts.index,target_counts.values,'Classes','Counts','Target class counts')
           
           #Plot missing values of features
           #missingValues(data)
           
           #print statistics of numeric data
           #printStat(data)
           
           #plot correlation of desired columns
           #colList = ['avgMins','callMinutesChangePct']
           #plotCorr(colList,data)
           
           #Plot filled and missing values
           #featureMisval(data['occupation']) 
           
           #Plot the distribution of categorical values
           #catCount('creditCard','churn',data)
           
           #Plot the density distribution of numerical data
           #numCount('handsetAge','churn',data)
           
           #Plot the density for combine features
           #combFeat('income','age','churn',data)
           
           #paralle coordinates plot
           #colList = ['avgBill','avgrecurringCharge','churn']
           #parallelPlot(colList,data)
           
           return features,target
           
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message


def missingValues(data):
    try:
           # Total missing values
           mis_val = data.isnull().sum()
         
           # Percentage of missing values
           mis_val_percent = 100 * mis_val / len(data)
           
           # Make a table with the results
           mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
           
           # Rename the columns
           mis_val_table_ren_columns = mis_val_table.rename(
           columns = {0 : 'Missing Values', 1 : '% of Total Values'})
           mis_val_table_ren_columns.head(4 )
           # Sort the table by percentage of missing descending
           misVal = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
                   '% of Total Values', ascending=False).round(1)
           
           # Print some summary information
           print ("Your selected dataframe has " + str(data.shape[1]) + " columns.\n"      
            "There are " + str(misVal.shape[0]) +
              " columns that have missing values.")
           
           print(mis_val_table_ren_columns.head(40))
                      
    except Exception as ex:
           print "-----------------------------------------------------------------------"
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print message
        
def featureMisval(feature):
    try:
           #check for spaces in the column occupation
           tser = feature
           ind = tser[tser == tser].index
           nind = tser[tser != tser].index
           tser.iloc[ind] = 'fill'
           tser.iloc[nind] = 'missing'
            
           plot_counts= tser.value_counts()                      
           barPlot(plot_counts.index,plot_counts.values,'Classes','Counts','Feature counts')           

    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message            
        
def barPlot(l1,l2,xd,yd,title):
    try:
        plt.figure(figsize=(10,5))
        sns.barplot(l1, l2, alpha=0.8)
        plt.title(title)
        plt.ylabel(yd, fontsize=12)
        plt.xlabel(xd, fontsize=12)
        plt.show()
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message

def catCount(feature,target,data): 
    try:
        d_f = data.loc[data[target] == False]
        d_t = data.loc[data[target] == True]
        
        d_f[feature].fillna(value='missing',inplace=True)
        d_t[feature].fillna(value='missing',inplace=True)
         
        f, axes = plt.subplots(1, 2, figsize=(8, 8), sharex=True)
        sns.countplot(x=feature, data=d_f,ax=axes[0])
        axes[0].set_title('churn=False')
        sns.countplot(x=feature, data=d_t,ax=axes[1])
        axes[1].set_title('churn=True')    
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=65, horizontalalignment='right')                
        plt.show()
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message

def numCount(feature,target,data):
    try:
        d_f = data.loc[data[target] == False]
        d_t = data.loc[data[target] == True]
         
        plt.figure(figsize = (12, 6))
        sns.kdeplot(d_f[feature], label='churn=false')
        sns.kdeplot(d_t[feature], label='churn=true')
        plt.title(feature)
        plt.legend();
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message        

def combFeat(f1,f2,target,data):
    col = ['combine',target]
    new_d = pd.DataFrame(columns = col)
    new_d['combine'] = data[f1]**2 + data[f2]**2
    new_d[target] = data[target]
    numCount('combine',target,new_d)

def parallelPlot(colList,data):
    try:
        td = pd.DataFrame(columns=colList)
        for col in colList:
            td[col] = data[col]
         
        pp.parallel_coordinates(td,'churn',color=('#556270','#4ECDC4'))
        plt.show()
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message         
    
def printStat(data):
    try:
#        col = ['customer','occupation','regionType','marriageStatus','children','smartPhone','creditRating','homeOwner','creditCard','churn']
#        td = data.drop(col,axis=1)
        td = data.drop(data.select_dtypes('object'))
        mins = td.min()
        maxs = td.max()
        means = td.mean()
        medians = td.median()
        stds = td.std()
        stats = pd.concat([mins,maxs,means,medians,stds], axis=1)
        stats.columns = ['Min','Max','Mean','Median','Std Dev']
        print(stats.head(25))
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message     
        
def plotCorr(colList,data):
    try:
        td = data[colList]
        f, ax = plt.subplots(figsize=(10, 6))
        corr = td.corr()
        sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm",fmt='.2f',linewidths=.05)
        f.subplots_adjust(top=0.93)
        f.suptitle('Churn Attributes Correlation Heatmap', fontsize=14)
    except Exception as ex:
        print "-----------------------------------------------------------------------"
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message        