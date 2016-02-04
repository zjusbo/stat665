""" Problem Set 01 starter code

Please make sure your code runs on Python version 3.5.0

Due date: 2016-02-05 13:00
"""
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt

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
    proba = nbrs.predict_proba(X_test)[:,1]
    return proba

def linearRegression(X_train, y_train, X_test):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    return y_predict, lr

def ridgeRegression(X_train, y_train, X_test):
    from sklearn.linear_model import RidgeCV
    rc = RidgeCV()
    rc.fit(X_train, y_train)
    y_predict = rc.predict(X_test)
    return y_predict, rc

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
        return proba

def q2():
    with open('nyc_train.csv', 'rb') as trainfile:
        trainObj = csv.DictReader(trainfile, delimiter = ',')
        x1_train = []
        x2_train = []
        y_train = []
        count = 0
        for row in trainObj:
            pickup_datetime = row['pickup_datetime']
            pickup_neighborhood = row['pickup_NTACode']
            head, sep, tail = pickup_datetime.partition(' ')
            hour, sep, minute_second = tail.partition(':')
            # convert NTACode to category
            hour = int(hour)
            itemY = int(row['dropoff_BoroCode'])
            if itemY != 1:
                itemY = 0
            x1_train.append(hour)
            x2_train.append(pickup_neighborhood)
            y_train.append(itemY)
            count += 1
        le = preprocessing.LabelEncoder()
        x2_train_label = x2_train
        x1_train = le.fit_transform(x1_train)
        x2_train = le.fit_transform(x2_train)
        labelmap = [' '] * 29
        for i in xrange(0, len(x2_train)):
            labelmap[x2_train[i]] = x2_train_label[i]
        print labelmap
        return 
        category1_max = max(x1_train) + 1
        category2_max = max(x2_train) + 1
        x1_category = np.zeros((len(x1_train), category1_max))
        x2_category = np.zeros((len(x2_train), category2_max))
        for i in xrange(0, len(x1_train)):
            x1_category[i, x1_train[i]] = 1
        for i in xrange(0, len(x2_train)):
            x2_category[i, x2_train[i]] = 1
        X_train = np.column_stack((x1_category, x2_category))
        X_train = X_train.astype(int)

    with open('nyc_test.csv', 'rb') as testfile:
        testObj = csv.DictReader(testfile, delimiter = ',')
        x1_test = []
        x2_test = []
        y_test = []
        for row in testObj:
            pickup_datetime = row['pickup_datetime']
            pickup_neighborhood = row['pickup_NTACode']
            head, sep, tail = pickup_datetime.partition(' ')
            hour, sep, minute_second = tail.partition(':')
            # convert NTACode to category
            hour = int(hour)
            x1_test.append(hour)
            x2_test.append(pickup_neighborhood)
        x1_test = le.fit_transform(x1_test)
        x2_test = le.fit_transform(x2_test)
        category1_max = max(x1_test) + 1
        category2_max = max(x2_test) + 1
        x1_category = np.zeros((len(x1_test), category1_max))
        x2_category = np.zeros((len(x2_test), category2_max))
        for i in xrange(0, len(x1_test)):
            x1_category[i, x1_test[i]] = 1
        for i in xrange(0, len(x2_test)):
            x2_category[i, x2_test[i]] = 1
        X_test = np.column_stack((x1_category, x2_category))
        X_test = X_test.astype(int)
        predict = linearRegression(X_train, y_train, X_test)
        return predict

def q3():
    with open('nyc_train.csv', 'rb') as trainfile:
        trainObj = csv.DictReader(trainfile, delimiter = ',')
        x1_train = []
        x2_train = []
        y_train = []
        count = 0
        for row in trainObj:
            pickup_datetime = row['pickup_datetime']
            pickup_neighborhood = row['pickup_NTACode']
            head, sep, tail = pickup_datetime.partition(' ')
            hour, sep, minute_second = tail.partition(':')
            # convert NTACode to category
            hour = int(hour)
            itemY = int(row['dropoff_BoroCode'])
            if itemY != 1:
                itemY = 0
            x1_train.append(hour)
            x2_train.append(pickup_neighborhood)
            y_train.append(itemY)
            count += 1
        le = preprocessing.LabelEncoder()
        
        x1_train = le.fit_transform(x1_train)
        x2_train = le.fit_transform(x2_train)
        category1_max = max(x1_train) + 1
        category2_max = max(x2_train) + 1
        x1_category = np.zeros((len(x1_train), category1_max))
        x2_category = np.zeros((len(x2_train), category2_max))
        for i in xrange(0, len(x1_train)):
            x1_category[i, x1_train[i]] = 1
        for i in xrange(0, len(x2_train)):
            x2_category[i, x2_train[i]] = 1
        X_train = np.column_stack((x1_category, x2_category))
        X_train = X_train.astype(int)

    with open('nyc_test.csv', 'rb') as testfile:
        testObj = csv.DictReader(testfile, delimiter = ',')
        x1_test = []
        x2_test = []
        y_test = []
        for row in testObj:
            pickup_datetime = row['pickup_datetime']
            pickup_neighborhood = row['pickup_NTACode']
            head, sep, tail = pickup_datetime.partition(' ')
            hour, sep, minute_second = tail.partition(':')
            # convert NTACode to category
            hour = int(hour)
            x1_test.append(hour)
            x2_test.append(pickup_neighborhood)
        x1_test = le.fit_transform(x1_test)
        x2_test = le.fit_transform(x2_test)
        category1_max = max(x1_test) + 1
        category2_max = max(x2_test) + 1
        x1_category = np.zeros((len(x1_test), category1_max))
        x2_category = np.zeros((len(x2_test), category2_max))
        for i in xrange(0, len(x1_test)):
            x1_category[i, x1_test[i]] = 1
        for i in xrange(0, len(x2_test)):
            x2_category[i, x2_test[i]] = 1
        X_test = np.column_stack((x1_category, x2_category))
        X_test = X_test.astype(int)

        predict, model = ridgeRegression(X_train, y_train, X_test)
        return predict, model

def gencsv():
    predict1 = q1()
    predict2 = q2()
    predict3 = q3()
    # write to csv file
    with open('result.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ',',
                            quotechar = '|', quoting = csv.QUOTE_MINIMAL)
        for i in xrange(0, len(predict1)):
            csvwriter.writerow([predict1[i], predict2[i], predict3[i]])

def plot():
    x1_unit = np.zeros((24, 24)).astype(int)
    x2_unit = np.zeros((29, 29)).astype(int)
    X_test = []
    for i in xrange(0, 29):
        x2_unit[i, i] = 1
    for i in xrange(0, 24):
        x1_unit[i, i] = 1
    for i in x1_unit:
        for j in x2_unit:
            X_test.append(list(i) + list(j))
    predict, model = q3()
    y_predict = model.predict(X_test)
    x1 = []
    y1 = []
    x0 = []
    y0 = []
    for i in xrange(0, len(X_test)):
        x = extractHour(X_test[i])
        y = extractNeighbor(X_test[i])
        if y_predict[i] > 0.9:
            x1.append(x)
            y1.append(y)
        else:
            x0.append(x)
            y0.append(y)
    plt.plot(x1, y1, 'ro')
    plt.plot(x0, y0, 'b^')
    plt.ylabel('neighbor code')
    plt.xlabel('hour')
    plt.show()

def extractHour(row):
    row = row[:24]
    for i in xrange(0, len(row)):
        if row[i] == 1:
            return i
def extractNeighbor(row):
    row = row[24:]
    for i in xrange(0, len(row)):
        if row[i] == 1:
            return i
def foo():
    plt.plot([1,2,3,4], [1,4,9,16], 'ro')
    plt.show()
if __name__ == "__main__":
    plot()