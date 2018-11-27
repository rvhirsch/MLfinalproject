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

    yTrain_bin = data.make_y_binary(yTrain)
    yTest_bin = data.make_y_binary(yTest)

    bin_train_error, bin_test_error = error.get_error_bin(xTrain, xTest, yTrain_bin, yTest_bin, LogisticRegression())
    multi_train_error, multi_test_error = error.get_error_multi(xTrain, xTest, yTrain, yTest, LogisticRegression())

    pd.options.display.float_format = '{:.2f}%'.format
    print (error.error_table(bin_train_error, bin_test_error, multi_train_error, multi_test_error))
