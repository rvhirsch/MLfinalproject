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

    bintrain = []
    bintest = []
    multitraineq = []
    multitesteq = []
    multitrain = []
    multitest = []

    Cvals = [0.01, 0.1, 1, 10, 100]
    for c in Cvals:
        print ("C:", c)
        bin_train_error, bin_test_error, multi_train_error_eq, multi_test_error_eq, multi_train_error, multi_test_error = error.compute_errors(xTrain, xTest, yTrain, yTest, LogisticRegression(C=c, solver='lbfgs'))

        bintrain.append(bin_train_error)
        bintest.append(bin_test_error)
        multitraineq.append(multi_train_error_eq)
        multitesteq.append(multi_test_error_eq)
        multitrain.append(multi_train_error)
        multitest.append(multi_test_error)

    error.graphCerrors(bintrain, bintest, multitraineq, multitesteq, multitrain, multitest, Cvals)
