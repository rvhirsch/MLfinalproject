import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import getdata as data  # getdata.py file
import errorcalc as error

from sklearn import svm

import warnings
warnings.filterwarnings("ignore") # suppress warnings

if __name__ == "__main__":
    fulldata = data.get_data("yelpdata.csv")
    # smalldata = fulldata[:100]
    # xTrain, xTest, yTrain, yTest = data.split_data(smalldata)

    xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

    C_Vals = [0.001, 0.01, 0.1, 1, 10]
    for c in C_Vals:
        print ("C value is :", c)
        error.compute_errors(xTrain, xTest, yTrain, yTest, svm.LinearSVC(C=c))
