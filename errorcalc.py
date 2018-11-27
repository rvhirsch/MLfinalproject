import pandas as pd
import numpy as np

def error_table(bin_in, bin_out, multi_in, multi_out):
    titles = ['Ein', 'Eout']
    binary = pd.Series([bin_in*100, bin_out*100], index=titles)
    multi = pd.Series([multi_in*100, multi_out*100], index=titles)
    d = {'Binary class' : binary, 'Multi-class' : multi}
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
