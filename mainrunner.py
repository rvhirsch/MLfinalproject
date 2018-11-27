import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import logisticregression as logreg
import perceptronclassifier as percep
import getdata as data

fulldata = data.get_data("yelpdata.csv")
# smalldata = fulldata[:100]
# xTrain, xTest, yTrain, yTest = data.split_data(smalldata)
xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

yTrain_bin = data.make_y_binary(yTrain)
yTest_bin = data.make_y_binary(yTest)

print ("Logistic Regression Error:")
bin_train_error, bin_test_error = logreg.get_log_error(xTrain, xTest, yTrain_bin, yTest_bin)
multi_train_error, multi_test_error = logreg.get_log_error(xTrain, xTest, yTrain, yTest)

pd.options.display.float_format = '{:.2f}%'.format
print (data.error_table(bin_train_error, bin_test_error, multi_train_error, multi_test_error))

print ("Perceptron Error:")
bin_train_error, bin_test_error = percep.get_per_error(xTrain, xTest, yTrain_bin, yTest_bin)
multi_train_error, multi_test_error = percep.get_per_error(xTrain, xTest, yTrain, yTest)

pd.options.display.float_format = '{:.2f}%'.format
print (data.error_table(bin_train_error, bin_test_error, multi_train_error, multi_test_error))
