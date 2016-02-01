""" Problem Set 01 starter code

Please make sure your code runs on Python version 3.5.0

Due date: 2016-02-05 13:00
"""
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

def my_knn(X_train, y_train, X_test, k=1):
    """ Basic k-nearest neighbor functionality

    k-nearest neighbor regression for a numeric test
    matrix. Prediction are returned for the same data matrix
    used for training. For each row of the input, the k
    closest rows (using the l2 distance) in the training
    set are identified. The mean of the observations y
    is used for the predicted value of a new observation.

    Args:
      X: an n by p numpy array; the data matrix of predictors
      y: a length n numpy array; the observed response
      k: integer giving the number of neighbors to include

    Returns:
      a 1d numpy array of predicted responses for each row of the input matrix X
    """
    X_train = np.array(X_train)
    nbrs = KNeighborsClassifier(n_neighbors = k)
    nbrs.fit(X_train, y_train)
    proba = nbrs.predict_proba(X_train)[:,1]
    return proba

def my_ksmooth(X_train, y_train, X_test, sigma=1.0):
    """ Kernel smoothing function

    kernel smoother for a numeric test matrix with a Gaussian
    kernel. Prediction are returned for the same data matrix
    used for training. For each row of the input, a weighted
    average of the input y is used for prediction. The weights
    are given by the density of the normal distribution for
    the distance of a training point to the input.

    Args:
      X: an n by p numpy array; the data matrix of predictors
      y: a length n numpy vector; the observed response
      sigma: the standard deviation of the normal density function
        used for the weighting scheme

    Returns:
      a 1d numpy array of predicted responses for each row of the input matrix X
    """
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    return y_predict

def ridgeRegression(X_train, y_train, X_test):
    from sklearn.linear_model import RidgeCV
    rc = RidgeCV()
    rc.fit(X_train, y_train)
    y_predict = rc.predict(X_test)
    return y_predict

def q1():
    with open('nyc_train.csv', 'rb') as trainfile:
        trainObj = csv.DictReader(trainfile, delimiter = ',')
        X_train = []
        y_train = []
        for row in trainObj:
            itemX = [float(row['pickup_longitude']), float(row['pickup_latitude'])]
            itemY = int(row['dropoff_BoroCode'])
            if itemY != 1:
                itemY = 0
            X_train.append(itemX)
            y_train.append(itemY)
    with open('nyc_test.csv', 'rb') as testfile:
        testObj = csv.DictReader(testfile, delimiter = ',')
        X_test = []
        for row in testObj:
            itemX = [float(row['pickup_longitude']), float(row['pickup_latitude'])]
            X_test.append(itemX)
        proba = my_knn(X_train, y_train, X_test, 100)
        print proba

def q2():
    import re
    with open('nyc_train.csv', 'rb') as trainfile:
        trainObj = csv.DictReader(trainfile, delimiter = ',')
        X_train = []
        y_train = []
        for row in trainObj:
            pickup_datetime = row['pickup_datetime']
            pickup_neighborhood = row['pickup_BoroCode']
            head, sep, tail = pickup_datetime.partition(' ')
            hour, sep, minute_second = tail.partition(':')
            pickup_neighborhood = int(pickup_neighborhood)
            hour = int(hour)
            itemY = int(row['dropoff_BoroCode'])
            if itemY != 1:
                itemY = 0
            X_train.append([hour, pickup_neighborhood])
            y_train.append(itemY)
    with open('nyc_test.csv', 'rb') as testfile:
        testObj = csv.DictReader(testfile, delimiter = ',')
        X_test = []
        for row in testObj:
            pickup_datetime = row['pickup_datetime']
            pickup_neighborhood = row['pickup_BoroCode']
            head, sep, tail = pickup_datetime.partition(' ')
            hour, sep, minute_second = tail.partition(':')
            pickup_neighborhood = int(pickup_neighborhood)
            hour = int(hour)
            X_test.append([hour, pickup_neighborhood])
        predict = my_ksmooth(X_train, y_train, X_test)
        print predict

def q3():
    import re
    with open('nyc_train.csv', 'rb') as trainfile:
        trainObj = csv.DictReader(trainfile, delimiter = ',')
        X_train = []
        y_train = []
        for row in trainObj:
            pickup_datetime = row['pickup_datetime']
            pickup_neighborhood = row['pickup_BoroCode']
            head, sep, tail = pickup_datetime.partition(' ')
            hour, sep, minute_second = tail.partition(':')
            pickup_neighborhood = int(pickup_neighborhood)
            hour = int(hour)
            itemY = int(row['dropoff_BoroCode'])
            if itemY != 1:
                itemY = 0
            X_train.append([hour, pickup_neighborhood])
            y_train.append(itemY)
    with open('nyc_test.csv', 'rb') as testfile:
        testObj = csv.DictReader(testfile, delimiter = ',')
        X_test = []
        for row in testObj:
            pickup_datetime = row['pickup_datetime']
            pickup_neighborhood = row['pickup_BoroCode']
            head, sep, tail = pickup_datetime.partition(' ')
            hour, sep, minute_second = tail.partition(':')
            pickup_neighborhood = int(pickup_neighborhood)
            hour = int(hour)
            X_test.append([hour, pickup_neighborhood])
            # dummy variable? 
            # check manual or implement it by hand
        y_predict = ridgeRegression(X_train, y_train, X_test)
        print y_predict   

def main():
    pass

start = 0
if __name__ == "__main__": 
   #  main()
   q3()
