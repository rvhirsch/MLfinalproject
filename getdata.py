import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def get_data(filename):
    data = pd.read_csv(filename, delimiter=",", header=0)
    return data

def standardize_data(data):
    scaler = preprocessing.StandardScaler()
    np_scaled = scaler.fit_transform(data)
    return np_scaled

def split_data(data):
    y = data.iloc[0:, -1] #labels
    X = data.iloc[0:, :-1] #features
    # X = standardize_data(X)

    # split 70% of data
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, shuffle=True)
    return xTrain, xTest, yTrain.values, yTest.values
