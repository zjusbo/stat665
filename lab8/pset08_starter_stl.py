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

X_train = np.genfromtxt('X_train_new.csv', delimiter=',')
Y_train = np.genfromtxt('Y_train.csv', delimiter=',')
X_test = np.genfromtxt('X_test_new.csv', delimiter=',')
Y_test = np.genfromtxt('Y_test.csv', delimiter=',')

# X_test = X_train[:1000]
# Y_test = Y_train[:1000]

# X_train = X_train[1000:2000]
# Y_train = Y_train[1000:2000]
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

	clf = svm.SVC(probability=True)
	clf.fit()
