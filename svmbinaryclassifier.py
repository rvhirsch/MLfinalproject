import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import getdata as data  # getdata.py file

from sklearn import svm
from sklearn.linear_model import LogisticRegression

def make_y_binary(yvals):
    yvals = np.where(yvals < 3, 1, -1)
    return yvals

def get_log_error(Xtrain, Xtest, ytrain, ytest):
    classifier = LogisticRegression()
    classifier.fit(Xtrain, ytrain)

    pred_train = classifier.predict(Xtrain)
    pred_test = classifier.predict(Xtest)
    # print (len(np.where(np.equal(y_pred, y_test))[0])/len(y_test))
    # print (np.sum(y_pred==y_test)/len(y_test))

    ein = np.sum(pred_train != ytrain)/len(ytrain)
    eout = np.sum(pred_test != ytest)/len(ytest)
    return ein, eout

def error_table(bin_in, bin_out, multi_in, multi_out):
    titles = ['Ein', 'Eout']
    binary = pd.Series([bin_in*100, bin_out*100], index=titles)
    multi = pd.Series([multi_in*100, multi_out*100], index=titles)
    d = {'Binary class' : binary, 'Multi-class' : multi}
    return pd.DataFrame(d)

fulldata = data.get_data("yelpdata.csv")
# smalldata = fulldata[:100]
# xTrain, xTest, yTrain, yTest = split_data(smalldata)
xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

yTrain_bin = make_y_binary(yTrain)
yTest_bin = make_y_binary(yTest)

bin_train_error, bin_test_error = get_log_error(xTrain, xTest, yTrain_bin, yTest_bin)

multi_train_error, multi_test_error = get_log_error(xTrain, xTest, yTrain, yTest)

pd.options.display.float_format = '{:.2f}%'.format
print (error_table(bin_train_error, bin_test_error, multi_train_error, multi_test_error))
