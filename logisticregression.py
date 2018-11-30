import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import getdata as data
import errorcalc as error

from sklearn import svm
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":

    fulldata = data.get_data("yelpdata.csv")
    # smalldata = fulldata[:100]
    # xTrain, xTest, yTrain, yTest = data.split_data(smalldata)

    xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

    Cvals = [0.01, 0.1, 1, 10, 100]
    for c in Cvals:
        print ("C:", c)
        error.compute_errors(xTrain, xTest, yTrain, yTest, LogisticRegression(C=c, solver='lbfgs'))
