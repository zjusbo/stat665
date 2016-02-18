#!/usr/bin/env python3
import csv
# A useful function for calculating the nearest neighbors:
#   neighbors.KNeighborsRegressor(n_neighbors)

# Use linear interpolation to predict new values on the test data
#   interp1d

# Save the results as "results.csv"
def main():
  # read train data
  filenames = ["ct_rac_S000_JT00_2013.csv", 
  "ny_rac_S000_JT00_2013.csv", 
  "ca_rac_S000_JT00_2013.csv",
  "mt_rac_S000_JT00_2013.csv"]

  print "loading ct..."
  X, y, ct_d = load(filenames[0])

  return 
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
  


  # print "training data..."
  # rtree = rfr(n_estimators = 79, max_features = 10, oob_score = True)    

  # rtree.fit(X, y)
  # print rtree.feature_importances_  

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
    for key, value in data.iteritems():      
      if key not in ['h_geocode', 'C000', 'CE01', 'CE02', 'CE03', 'createdate']:
        print key
    # X stores the feature matrix
    # construct y
    
    return X, y, data


def save(x):
  with open("results.csv", 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter = ',')
    print x
    for row in x:
      csvwriter.writerow([row])

if __name__ == '__main__':
  main()