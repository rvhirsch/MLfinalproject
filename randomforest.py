import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import getdata as data  # getdata.py file
import errorcalc as error

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore") # suppress warnings

if __name__ == "__main__":
    fulldata = data.get_data("yelpdata.csv")
    # smalldata = fulldata[:100]
    # xTrain, xTest, yTrain, yTest = data.split_data(smalldata)

    xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

    max_leaf_nodes = [6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    for leaf in max_leaf_nodes:
        print ("leaf:", leaf)
        error.compute_errors(xTrain, xTest, yTrain, yTest, RandomForestClassifier(max_leaf_nodes=leaf))
