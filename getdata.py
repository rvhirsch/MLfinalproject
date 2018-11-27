import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def get_data(filename):
    data = pd.read_csv(filename, delimiter=",", header=0)
    return data

def standardize_data(data):
    # scaler = preprocessing.StandardScaler()
    # np_scaled = scaler.fit_transform(data)
    np_scaled = preprocessing.scale(data)
    return np_scaled

def make_y_binary(yvals):
    yvals = np.where(yvals < 3, 1, -1)
    return yvals

def split_data(data):
    y = data.iloc[0:, -1] #labels
    X = data.iloc[0:, :-1] #features
    X = standardize_data(X)

    # split 70% of data
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, shuffle=True)
    return xTrain, xTest, yTrain.values, yTest.values
