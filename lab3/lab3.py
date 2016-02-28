#python3
import numpy as np
from scipy.stats.stats import pearsonr

global le_loc, enc_loc, le_beat, enc_beat, le_dist, enc_dist

def load(filename):
  import csv
  import math
  from sklearn.preprocessing import OneHotEncoder, LabelEncoder
  with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter = '|')
    data = []
    for row in reader:
      # ignore first line
      if row[0] == 'type':
        continue
      data.append(row)
    data = np.array(data)
  
    y = data[:, 0].astype(int)
    
    year = data[:, 1].astype(int).reshape(-1, 1)
    month = data[:, 2].astype(int).reshape(-1, 1)
    hour = data[:, 3].astype(int).reshape(-1, 1)
   # time = toTimestamp(year, month, hour)
   # time = time.reshape(-1, 1)
    
    arrest = data[:, 4].astype(int).reshape(-1, 1)
    domestic = data[:, 5].astype(int).reshape(-1, 1)

    global le_loc, enc_loc, le_beat, enc_beat, le_dist, enc_dist
    
    le_loc = LabelEncoder()
    le_beat = LabelEncoder()
    le_dist = LabelEncoder()
    enc_loc = OneHotEncoder()
    enc_beat = OneHotEncoder()
    enc_dist = OneHotEncoder()

    loc = le_loc.fit_transform(data[:, 6]).reshape(-1,1) # label loc
    enc_loc.fit(loc)
    loc = enc_loc.transform(loc).toarray()
    
    beat = le_beat.fit_transform(data[:,7]).reshape(-1,1) # label beat
    enc_beat.fit(beat)
    beat = enc_beat.transform(beat).toarray()

    dist = le_dist.fit_transform(data[:,8]).reshape(-1,1) # label dist
    enc_dist.fit(dist)
    dist = enc_dist.transform(dist).toarray()

    #X = np.hstack((year, month, hour, arrest, domestic, loc))
    X = np.hstack((year, month, hour, arrest, domestic, loc, beat, dist))
    return X, y


def toTimestamp(year, month, hour):
  return np.array(year * 12 * 30 * 24 + month * 30 * 24 + hour)

def pca(X):
  from sklearn.preprocessing import normalize
  from sklearn.decomposition import PCA
  import matplotlib.pyplot as plt
  #p = PCA()

  p = PCA(n_components=0.99) # keep 99% variance
  #X = normalize(X, axis = 0)
  p.fit(X)
  
  #plt.plot(p.explained_variance_ratio_[:10])
  #plt.show()
  #print(p.explained_variance_ratio_) 
  return p

def fit(X, y):
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier(n_estimators=50, verbose = 4, max_features = "auto")
   rf.fit(X, y)
   return rf

def loadTest(filename):
  import csv
  import math
  from sklearn.preprocessing import OneHotEncoder, LabelEncoder
  with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter = '|')
    data = []
    for row in reader:
      # ignore first line
      if row[0] == 'type':
        continue
      #if row[0] != 'NA':
      data.append(row)
    data = np.array(data)
    
    #y = data[:, 0].astype(int)
    y = []
    year = data[:, 1].astype(int).reshape(-1, 1)
    month = data[:, 2].astype(int).reshape(-1, 1)
    hour = data[:, 3].astype(int).reshape(-1, 1)
    time = toTimestamp(year, month, hour)
    time = time.reshape(-1, 1)
    
    arrest = data[:, 4].astype(int).reshape(-1, 1)
    domestic = data[:, 5].astype(int).reshape(-1, 1)

    loc = le_loc.transform(data[:, 6]).reshape(-1,1) # label loc
    loc = enc_loc.transform(loc).toarray()

    beat = le_beat.transform(data[:,7]).reshape(-1,1) # label beat
    beat = enc_beat.transform(beat).toarray()

    dist = le_dist.transform(data[:,8]).reshape(-1,1) # label dist
    dist = enc_dist.transform(dist).toarray()


    #X = np.hstack((year, month, hour, arrest, domestic, loc))
    X = np.hstack((year, month, hour, arrest, domestic, loc, beat, dist))
    return X, y

def svm_fit(X, y):
  from sklearn import svm
  clf = svm.SVC(decision_function_shape='ovo')
  clf.fit(X, y)
  return clf


def main():
  print("loading training set..")
  X, y = load('chiCrimeTrain.psv')
  print("loading test set...")
  X_test, y_test = loadTest('chiCrimeTest.psv')
  # for i in range(0, 10):
  #  print (X[i,:])
  print("performing pca...")
  p = pca(X)
  re_X = p.transform(X)
  #re_X = X
  print(re_X.shape)
  print("fitting tree..")
  tree = fit(re_X, y)
  #svm = svm_fit(re_X, y)
  re_X_test = p.transform(X_test)
# re_X_test = X_test
  print("predicting...")
  y_predict = tree.predict(re_X_test)
  #y_predict = svm.predict(re_X_test)
  y_predict = np.array(y_predict)
  #mis_classification_rate = float(np.where(y_predict != y_test)[0].size) / len(y_predict)
  save(y_predict)
  print('finish')
  #print (mis_classification_rate)

def confusion():
  import csv
  from sklearn.metrics import confusion_matrix
  with open('chiCrimeTest.psv', 'r') as f:
    reader = csv.reader(f, delimiter = '|')
    test_y = []
    order = []
    for idx, row in enumerate(reader):
      y = row[0]
      if y != 'NA' and y != 'type':
        order.append(idx - 1)
        test_y.append(y)
  order = np.array(order)
  test_y = np.array(test_y).astype(int)

  with open('pset03.csv', 'r') as f:
    reader = csv.reader(f, delimiter = '|')
    predict_y = []
    for idx, row in enumerate(reader):
      y = row[0]
      predict_y.append(y)
    predict_y = np.array(predict_y)
    predict_y = predict_y[order].astype(int)
    print (test_y)
    print (predict_y)
    #cm = confusion_matrix(test_y, predict_y)
    #print (float(len(np.where(predict_y == test_y))) / predict_y.shape[0])
    cm = confusion_matrix(test_y, predict_y, labels = [1,2,3,4,5])
    print(cm)

def save(x):
    import csv
    with open("pset03.csv", 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile, delimiter = ',')
      for row in x:
        csvwriter.writerow([row])

if __name__ == '__main__':
  confusion()