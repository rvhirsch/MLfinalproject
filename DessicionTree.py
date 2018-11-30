import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import getdata as data  # getdata.py file
import errorcalc as error

from sklearn import svm
from sklearn import tree

import warnings
warnings.filterwarnings("ignore") # suppress warnings

if __name__ == "__main__":
    fulldata = data.get_data("yelpdata.csv")
    # smalldata = fulldata[:100]
    # xTrain, xTest, yTrain, yTest = data.split_data(smalldata)

    xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

    vals = [10,  11, 12, 13, 14,15, 16, 17, 18, 19, 20]
    for v in vals:
        print ("v:", v)
        error.compute_errors(xTrain, xTest, yTrain, yTest, tree.DecisionTreeClassifier(max_depth=v))
