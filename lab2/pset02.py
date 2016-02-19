#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import interp1d

import sys
import csv
# A useful function for calculating the nearest neighbors:
#   neighbors.KNeighborsRegressor(n_neighbors)

# Use linear interpolation to predict new values on the test data
#   interp1d



class additivemodel(object):
  """docstring for additivemodel"""
  def __init__(self):
    super(additivemodel, self).__init__()
    self.isTrained = False
    self.g = []
    self.n = 0
    self.p = 0
    self.X = []
    self.interp1dmodel = []

  def fit(self, X, y):
    # convert list to np array
    X = np.array(X).astype(float)
    y = np.array(y).astype(float)  

    # x matrix, for inerpolation
    self.X = X
    # g matrix, init to 0.0
    self.g = np.zeros(X.shape)

    # n: number of samples 
    self.n = X.shape[0]

    # p: number of variables
    self.p = X.shape[1]

    # alpha_hat, avg of y
    self.alpha_hat = np.average(y)

    # iterate for 25 times
    for i in range(0, 25):
      # for every variable
      for j in range(0, self.p):
        r_j = y - self.alpha_hat - (np.sum(self.g, 1) - self.g[:, j])
        self.g[:, j] = self.smooth(X[:, j], r_j) 
        self.g[:, j] = self.g[:, j] - np.average(self.g[:, j])
    
    # backfiting algorithm end
    # g matrix is ready 
    self.isTrained = True
    self._interpolate()


  # gen interpolate model 
  def _interpolate(self):
    # for every variable j
    for j in range(0, self.p):
      col_x = self.X[:,j]
      col_y = self.g[:,j]
      order = col_x.argsort()
      self.interp1dmodel.append(interp1d(col_x[order], col_y[order]))

  def predict(self, X):
    # use model g to predict on X
    # for every variable j
    X = np.array(X).astype(float)
    
    # 1 * n, row np vector
    predict_y = np.zeros((1, X.shape[0]))
    for j in range(0, self.p):
      # 1 * n, row list vector
      g = self.interp1dmodel[j](X[:,j])
      predict_y += g
    predict_y += self.alpha_hat
    return predict_y[0]

  def smooth(self, X, y):
    # KNN algorithm for smooth
    nbrs = KNeighborsRegressor(n_neighbors = 20)
    X = X.reshape(-1, 1)
    nbrs.fit(X, y)
    proba = nbrs.predict(X)
    return proba

# Save the results as "results.csv"
def main():
  if len(sys.argv) < 3:
    print("Usage: <training file> <test file>")
    return 
  trainfile = sys.argv[1]
  testfile = sys.argv[2]
  train_y = []
  train_X = []
  
  # read train data
  with open(trainfile, 'r') as csvfile:
    trainreader = csv.reader(csvfile, delimiter = ',')
    for row in trainreader:
       y = row[0]

       X = row[1:] 

       train_y.append(y)
       train_X.append(X)
  test_X = []
  
  # read test data
  with open(testfile, 'r') as csvfile:
    testreader = csv.reader(csvfile, delimiter = ',')
    for row in testreader:
       X = row 
       test_X.append(X)
  model = additivemodel()
  model.fit(train_X, train_y)
  predict_y = model.predict(test_X)
  
  # save result
  save(predict_y)  

def save(x):
  with open("results.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter = ',')
    for row in x:
      csvwriter.writerow([row])

if __name__ == '__main__':
  main()