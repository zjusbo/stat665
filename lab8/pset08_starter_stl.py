import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn import svm

# load the STL-10 crime data into Python. you need to first
# download these from here:
#    http://euler.stat.yale.edu/~tba3/class_data/stl10
test = False

X_train = np.genfromtxt('X_train_new.csv', delimiter=',')
Y_train = np.genfromtxt('Y_train.csv', delimiter=',')
if test == False:
	X_test = np.genfromtxt('X_test_new.csv', delimiter=',')
	Y_test = np.genfromtxt('Y_test.csv', delimiter=',')
if test == True:
	X_test = X_train[:1000]
	Y_test = Y_train[:1000]

	X_train = X_train[1000:2000]
	Y_train = Y_train[1000:2000]

def part1_nn():
	model = Sequential()
	model.add(Dense(1000, input_shape=(4096,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer=RMSprop())

	model.fit(X_train, Y_train, batch_size=32, nb_epoch=25,
	            verbose=1, show_accuracy=True,
	            validation_split=0.2,
	            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
	print("Classification rate %02.5f" % (model.evaluate(X_test, Y_test, show_accuracy=True)[1]))

def part1_svm():
	global Y_train, Y_test
	clf = svm.SVC(decision_function_shape='ovo')
	Y_train = np.dot(Y_train, np.array([0,1,2,3,4,5,6,7,8,9])) # convert one-hot encode to 1 column
	Y_test = np.dot(Y_test, np.array([0,1,2,3,4,5,6,7,8,9])) # convert one-hot encode to 1 column
	clf.fit(X_train, Y_train)
	print("Mean accuracy: %02.5f" % (clf.score(X_test, Y_test)))

if __name__ == '__main__':
	part1_svm()