import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import getdata as data  # getdata.py file

from sklearn import svm

import warnings
warnings.filterwarnings("ignore") # suppress warnings

def get_svc_error(Xtrain, Xtest, ytrain, ytest):
    classifier = svm.LinearSVC()
    classifier.fit(Xtrain, ytrain)

    pred_train = classifier.predict(Xtrain)
    pred_test = classifier.predict(Xtest)
    # print (len(np.where(np.equal(y_pred, y_test))[0])/len(y_test))
    # print (np.sum(y_pred==y_test)/len(y_test))

    ein = np.sum(pred_train != ytrain)/len(ytrain)
    eout = np.sum(pred_test != ytest)/len(ytest)
    return ein, eout

if __name__ == "__main__":
    fulldata = data.get_data("yelpdata.csv")
    # smalldata = fulldata[:100]
    # xTrain, xTest, yTrain, yTest = data.split_data(smalldata)
    xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

    yTrain_bin = data.make_y_binary(yTrain)
    yTest_bin = data.make_y_binary(yTest)

    bin_train_error, bin_test_error = get_svc_error(xTrain, xTest, yTrain_bin, yTest_bin)
    multi_train_error, multi_test_error = get_svc_error(xTrain, xTest, yTrain, yTest)

    pd.options.display.float_format = '{:.2f}%'.format
    print (data.error_table(bin_train_error, bin_test_error, multi_train_error, multi_test_error))
