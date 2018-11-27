import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import getdata as data  # getdata.py file
import errorcalc as error

from sklearn import svm
from sklearn.linear_model import Perceptron

import warnings
warnings.filterwarnings("ignore") # suppress warnings

if __name__ == "__main__":
    fulldata = data.get_data("yelpdata.csv")
    # smalldata = fulldata[:100]
    # xTrain, xTest, yTrain, yTest = data.split_data(smalldata)
    xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

    yTrain_bin = data.make_y_binary(yTrain)
    yTest_bin = data.make_y_binary(yTest)

    bin_train_error, bin_test_error = error.get_log_error_bin(xTrain, xTest, yTrain_bin, yTest_bin, Perceptron())
    multi_train_error, multi_test_error = error.get_log_error_multi(xTrain, xTest, yTrain, yTest, Perceptron())

    pd.options.display.float_format = '{:.2f}%'.format
    print (error.error_table(bin_train_error, bin_test_error, multi_train_error, multi_test_error))
