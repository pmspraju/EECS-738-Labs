# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:54:50 2020

@author: pmspr
"""
#Step 0: Import relevant packages


#Step 1: Load imdb database
from projectFunctions import loadData
x_train, y_train, x_test, y_test = loadData()

#Step 2: Pad train and test data
from projectFunctions import padInput
x_train, x_test = padInput(x_train, x_test)

#Step 3: Create a 1D CNN for baseline
from projectFunctions import cnn11D
cnn11D(x_train, x_test, y_train, y_test)