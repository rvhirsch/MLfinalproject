import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import logisticregression as logreg
import perceptronclassifier as percep
import getdata as data
import linearsvc as svc
import errorcalc as error

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

import warnings
warnings.filterwarnings("ignore") # suppress warnings

fulldata = data.get_data("yelpdata.csv")
# smalldata = fulldata[:100]
# xTrain, xTest, yTrain, yTest = data.split_data(smalldata)
xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

# yTrain_bin = data.make_y_binary(yTrain)
# yTest_bin = data.make_y_binary(yTest)

print ("Logistic Regression Error:")
error.compute_errors(xTrain, xTest, yTrain, yTest, LogisticRegression())

print ("Perceptron Error:")
error.compute_errors(xTrain, xTest, yTrain, yTest, Perceptron())

print ("Linear SVC Error:")
error.compute_errors(xTrain, xTest, yTrain, yTest, svm.LinearSVC())
