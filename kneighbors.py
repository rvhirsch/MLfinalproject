import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import getdata as data  # getdata.py file
import errorcalc as error

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore") # suppress warnings

if __name__ == "__main__":
    fulldata = data.get_data("yelpdata.csv")
    # smalldata = fulldata[:100]
    # xTrain, xTest, yTrain, yTest = data.split_data(smalldata)

    xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

    leaves = [2, 5, 10, 15, 20]
    for leaf in leaves:
        print ("leaf:", leaf)
        error.compute_errors(xTrain, xTest, yTrain, yTest, KNeighborsClassifier(n_neighbors=leaf))
