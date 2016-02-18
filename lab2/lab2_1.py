#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor as rfr


from sklearn.linear_model import LinearRegression
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
    X = np.array(X)
    y = np.array(y)  

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
    for i in xrange(0, 25):
      # for every variable
      for j in xrange(0, self.p):
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
    for j in xrange(0, self.p):
      col_x = self.X[:,j]
      col_y = self.g[:,j]
      order = col_x.argsort()
      self.interp1dmodel.append(interp1d(col_x[order], col_y[order]))

  def predict(self, X):
    # use model g to predict on X
    # for every variable j
    X = np.array(X)
    
    # 1 * n, row np vector
    predict_y = np.zeros((1, X.shape[0]))
    for j in xrange(0, self.p):
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



def getmse(y1, y2):
  diff = np.array(y1) - np.array(y2)
  MSE = np.sum(np.multiply(diff, diff)) / diff.shape[0]
  return MSE
# Save the results as "results.csv"
def main():
  # read train data
  filenames = ["ct_rac_S000_JT00_2013.csv", 
  "ny_rac_S000_JT00_2013.csv", 
  "ca_rac_S000_JT00_2013.csv",
  "mt_rac_S000_JT00_2013.csv"]

  print "loading ct..."
  X, y, ct_d = load(filenames[0])

  
  # print "loading ny..."
  # ny_X, ny_y, ny_d = load(filenames[1])
  # print "loading ca..."
  # ca_X, ca_y, ca_d = load(filenames[2])
  # print "loading mt..."
  # mt_X, mt_y, mt_d = load(filenames[3])

    # lr = LinearRegression()
    # lr.fit(X, y)
    # predict_y = lr.predict(X)
    
    # print "ct: ", getmse(y, predict_y)
    
    # predict_y = lr.predict(ny_X)
    # print "ny: ", getmse(ny_y, predict_y)
    
    # predict_y = lr.predict(ca_X)
    # print "ca: ", getmse(ca_y, predict_y)
    
    # predict_y = lr.predict(mt_X)
    # print "mt: ", getmse(mt_y, predict_y)
  


  print "training data..."
  rtree = rfr(n_estimators = 79, max_features = 10, oob_score = True)    

  rtree.fit(X, y)
  print rtree.feature_importances_  

  #   # MSE = getmse(rtree.oob_prediction_, y)
  #   # print "Error for ct: ", MSE
    
  #   # rtree.fit(ny_X, ny_y)
  #   # MSE = getmse(rtree.oob_prediction_, ny_y)
  # # rtree.fit(ca_X, ca_y)
  # h_geocode = ny_d['h_geocode']
  # for i in xrange(0, len(h_geocode)):
  #   h_geocode[i] = h_geocode[i][2:5]
  # uni_geocode = set(h_geocode)
  # h_geocode = np.array(h_geocode)
  # ny_y = np.array(ny_y)
  
  # predict_y = rtree.predict(ny_X)
  # predict_y = np.array(predict_y)

  # mses = []
  # for item in uni_geocode:
  #   idx = np.where(h_geocode == item)
  #   MSE = getmse(predict_y[idx], ny_y[idx])
  #   mses.append([MSE, item])
  # mses = sorted(mses)
  # maxmses = mses[-10:]
  # print "ny.."
  # print maxmses
  
  # print "dealing with ca..."
  # h_geocode = ca_d['h_geocode']
  # for i in xrange(0, len(h_geocode)):
  #   h_geocode[i] = h_geocode[i][2:5]
  # uni_geocode = set(h_geocode)
  # h_geocode = np.array(h_geocode)
  # ca_y = np.array(ca_y)
  
  # predict_y = rtree.predict(ca_X)
  # predict_y = np.array(predict_y)

  # print 'calculating MSE for ca...'
  # mses = []
  # for item in uni_geocode:
  #   idx = np.where(h_geocode == item)
  #   MSE = getmse(predict_y[idx], ca_y[idx])
  #   mses.append([MSE, item])
  # mses = sorted(mses)
  # maxmses = mses[-10:]
  # print "ca.."
  # print maxmses


    
  # rtree.fit(mt_X, mt_y)
  # MSE = getmse(rtree.oob_prediction_, mt_y)
  # print "Error for ct: ", MSE
  


def load(filename):
  with open(filename, 'rb') as csvfile:
    reader = csv.DictReader(csvfile, delimiter = ',')
    data = {}
    
    # init key of data
    for row in reader:
      for key in row.iterkeys():
        data[key] = []
      break

    # read data
    for row in reader:
       for key, value in row.iteritems():
        data[key].append(value)

    X = []
    y = []
    C000 = np.array(data['C000']).astype(float)
    for key, value in data.iteritems():      
      if key not in ['h_geocode', 'C000', 'CE01', 'CE02', 'CE03', 'createdate']:
        # convert to np array
        v = np.array(value).astype(int)
        v = np.divide(v, C000)
        v = v.reshape((-1,1))
        if X == []: # first column
          X = v
        else:
          X = np.hstack((X, v))

    # X stores the feature matrix
    # construct y
    CE03 = np.array(data['CE03']).astype(float)
    y = np.array(np.divide(CE03, C000))
    
    return X, y, data


def save(x):
  with open("results.csv", 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter = ',')
    print x
    for row in x:
      csvwriter.writerow([row])

if __name__ == '__main__':
  main()