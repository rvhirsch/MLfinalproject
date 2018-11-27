import pandas as pd
import numpy as np

import getdata as data

def compute_errors(xTrain, xTest, yTrain, yTest, classifier):
    yTrain_bin = data.make_y_binary(yTrain)
    yTest_bin = data.make_y_binary(yTest)

    bin_train_error, bin_test_error = get_error_bin(xTrain, xTest, yTrain_bin, yTest_bin, classifier)
    multi_train_error_eq, multi_test_error_eq = get_error_bin(xTrain, xTest, yTrain, yTest, classifier)
    multi_train_error, multi_test_error = get_error_multi(xTrain, xTest, yTrain, yTest, classifier)

    pd.options.display.float_format = '{:.2f}%'.format
    print (error_table(bin_train_error, bin_test_error,
                        multi_train_error_eq, multi_test_error_eq,
                        multi_train_error, multi_test_error))

def error_table(bin_in, bin_out, multi_in_eq, multi_out_eq, multi_in_one_off, multi_out_one_off):
    titles = ['Ein', 'Eout']
    binary = pd.Series([bin_in*100, bin_out*100], index=titles)
    multi_eq = pd.Series([multi_in_eq*100, multi_out_eq*100], index=titles)
    multi_one = pd.Series([multi_in_one_off*100, multi_out_one_off*100], index=titles)
    d = {'Binary class' : binary, 'Multi-class (==)' : multi_eq, 'Multi-class (+/- 1)' : multi_one}
    return pd.DataFrame(d)

def equal_error(pred_train, ytrain, pred_test, ytest):
    ein = np.sum(pred_train != ytrain)/len(ytrain)
    eout = np.sum(pred_test != ytest)/len(ytest)
    return ein, eout

def off_by_one_err(pred_train, ytrain, pred_test, ytest): # only for multiclass
    ein = np.sum(pred_train < ytrain - 1)
    ein += np.sum(pred_train > ytrain + 1)
    ein /= len(ytrain)
    # eout = np.sum((pred_test < ytest - 1) or (pred_test > ytest + 1))/len(ytest)
    eout = np.sum(pred_test < ytest - 1)
    eout += np.sum(pred_test > ytest + 1)
    eout /= len(ytrain)
    return ein, eout


def get_error_bin(Xtrain, Xtest, ytrain, ytest, classifier):
    classifier.fit(Xtrain, ytrain)

    pred_train = classifier.predict(Xtrain)
    pred_test = classifier.predict(Xtest)

    return equal_error(pred_train, ytrain, pred_test, ytest)

def get_error_multi(Xtrain, Xtest, ytrain, ytest, classifier):
    classifier.fit(Xtrain, ytrain)

    pred_train = classifier.predict(Xtrain)
    pred_test = classifier.predict(Xtest)

    return off_by_one_err(pred_train, ytrain, pred_test, ytest)
