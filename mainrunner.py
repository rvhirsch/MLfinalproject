import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import logisticregression as logreg
import perceptronclassifier as percep
import getdata as data
import linearsvc as svc
import errorcalc as error

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

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
error.compute_errors(xTrain, xTest, yTrain, yTest, Perceptron(max_iter=100))

print ("MLP Error:")
error.compute_errors(xTrain, xTest, yTrain, yTest, MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,5)))

print ("Random Forest Error:")
error.compute_errors(xTrain, xTest, yTrain, yTest, RandomForestClassifier(max_leaf_nodes=7500))

print ("K Neighbors Error:")
error.compute_errors(xTrain, xTest, yTrain, yTest, KNeighborsClassifier(leaf_size=5, n_neighbors=15))

print ("K Means Error:")
error.compute_errors(xTrain, xTest, yTrain, yTest, KMeans(n_clusters=5))

print ("Linear SVC Error:")
error.compute_errors(xTrain, xTest, yTrain, yTest, svm.LinearSVC())
