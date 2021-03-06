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

    iters = [10, 20, 50, 100, 200, 300]
    for i in iters:
        print ("i:", i)
        error.compute_errors(xTrain, xTest, yTrain, yTest, Perceptron(max_iter=i))
