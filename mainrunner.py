import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import logisticregression as logreg
import perceptronclassifier as percep
import getdata as data
import linearsvc as svc
import errorcalc as error

fulldata = data.get_data("yelpdata.csv")
# smalldata = fulldata[:100]
# xTrain, xTest, yTrain, yTest = data.split_data(smalldata)
xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

yTrain_bin = data.make_y_binary(yTrain)
yTest_bin = data.make_y_binary(yTest)

print ("Logistic Regression Error:")
bin_train_error, bin_test_error = error.get_log_error_bin(xTrain, xTest, yTrain_bin, yTest_bin, LogisticRegression())
multi_train_error, multi_test_error = error.get_log_error_multi(xTrain, xTest, yTrain, yTest, LogisticRegression())

pd.options.display.float_format = '{:.2f}%'.format
print (error.error_table(bin_train_error, bin_test_error, multi_train_error, multi_test_error))

print ("Perceptron Error:")
bin_train_error, bin_test_error = error.get_log_error_bin(xTrain, xTest, yTrain_bin, yTest_bin, Perceptron())
multi_train_error, multi_test_error = error.get_log_error_multi(xTrain, xTest, yTrain, yTest, Perceptron())

pd.options.display.float_format = '{:.2f}%'.format
print (data.error_table(bin_train_error, bin_test_error, multi_train_error, multi_test_error))

print ("Linear SVC Error:")
bin_train_error, bin_test_error = error.get_log_error_bin(xTrain, xTest, yTrain_bin, yTest_bin, svm.LinearSVC())
multi_train_error, multi_test_error = error.get_log_error_multi(xTrain, xTest, yTrain, yTest, svm.LinearSVC())

pd.options.display.float_format = '{:.2f}%'.format
print (data.error_table(bin_train_error, bin_test_error, multi_train_error, multi_test_error))
