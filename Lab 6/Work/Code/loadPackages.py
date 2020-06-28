# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:53:59 2020

@author: pmspr
"""
#Import relevant packages
from __future__ import print_function

import os
import sys
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.activations import relu
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

from tensorflow.keras.datasets import imdb

print("_"*100)
print(os.getcwd())
print("Modules imported \n")
print("Files in current directory:")
from subprocess import check_output
#print(check_output(["ls", "../data"]).decode("utf8")) #check the files available in the directory

print("Packages Loaded")
print('The Tensorflow version is {}.'.format(tf.__version__))
print('The Keras version is {}.'.format(keras.__version__))
print('The Pandas version is {}.'.format(pd.__version__))
print('The Numpy version is {}.'.format(np.__version__))
print(np.__file__)
print("_"*100)