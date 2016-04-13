import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.utils import np_utils


# set this to false once you have tested your code!
TEST = False
# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

data_augmentation = False

# function to read in and process the cifar-10 data; set the
# number of classes you want
def load_data(nclass):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # down-sample to three classes
    X_train = X_train[(y_train < nclass).reshape(50000)]
    y_train = y_train[(y_train < nclass).reshape(50000)]
    X_test = X_test[(y_test < nclass).reshape(10000)]
    y_test = y_test[(y_test < nclass).reshape(10000)]
    # create responses
    Y_train = np_utils.to_categorical(y_train, nclass)
    Y_test = np_utils.to_categorical(y_test, nclass)
    if TEST:
        X_train = X_train[:1000]
        Y_train = Y_train[:1000]
        X_test = X_test[:1000]
        Y_test = Y_test[:1000]
    return X_train, Y_train, X_test, Y_test



X_train, Y_train, X_test, Y_test = load_data(2)
# Note: You'll need to do this manipulation to construct the
# output of the autoencoder. This is because the autoencoder
# will have a flattend dense layer on the output, and you need
# to give Keras a flatted version of X_train
X_train_auto_output = X_train.reshape(X_train.shape[0], 3072)
X_test_auto_output = X_test.reshape(X_test.shape[0], 3072)

def copy_freeze_model(model, nlayers = 1):
    new_model = Sequential()
    for l in model.layers[:nlayers]:
        l.trainable = False
        new_model.add(l)
    return new_model

def part1(size):
    # Convolution model kernel size
    # 2D convolution
    model = Sequential()

    model.add(Convolution2D(32, size, size, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model = train(model, auto=False)    

    print("Kernel size: %d * %d, Classification rate %02.5f" % (size, size, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))
    return model

def part2(size):
    # Convolution model kernel size
    # Autoencoder
    model = Sequential()

    model.add(Convolution2D(32, size, size, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(img_channels * img_rows * img_cols))
    
    model = train(model, auto=True)

    print("Kernel size: %d * %d, MSE %02.5f" % (size, size, model.evaluate(X_test, X_test_auto_output, show_accuracy=True)[0]))
    return model

def train(model, auto):
    if auto == True:
        model.compile(loss='mean_squared_error',
              optimizer=RMSprop())

        model.fit(X_train, X_train_auto_output, batch_size=32, nb_epoch=25,
            verbose=1, show_accuracy=True,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
            )
    else:
        model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop())

        model.fit(X_train, Y_train, batch_size=32, nb_epoch=25,
            verbose=1, show_accuracy=True,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
        
    return model


def add_top_layer(model):
    # add top dense layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model
    
def part3():
    size = 3
    # Convolution model kernel size
    # Frezzing layer
    model = part1(size)
    # copy freeze model 
    # freeze convolutional layer
    model = copy_freeze_model(model, 4)
    model = add_top_layer(model)
    model = train(model, auto=False)

    print("Kernel size: %d * %d, Classification rate %02.5f" % (size, size, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))
    model.save_weights('part3_1.h5')

    model = part2(size)
    model = copy_freeze_model(model, 4)
    model = add_top_layer(model)
    model = train(model, auto=False)
    model.save_weights('part3_2.h5')
    print("Kernel size: %d * %d, Classification rate %02.5f" % (size, size, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))

def part4(weights):
    global X_train, Y_train, X_test, Y_test
    size = 3
    model = Sequential()

    model.add(Convolution2D(32, size, size, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model = add_top_layer(model)

    model.load_weights(weights)

    model = copy_freeze_model(model, 4)

    # add top dense layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    

    X_train, Y_train, X_test, Y_test = load_data(10)

    model = train(model, auto=False)
    print("Classification rate %02.5f" % (model.evaluate(X_test, Y_test, show_accuracy=True)[1]))

def part5():
    size = 3
    model = Sequential()

    model.add(Convolution2D(32, size, size, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, size, size))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model = add_top_layer(model)

    model = train(model, auto=False)

    print("Classification rate %02.5f" % (model.evaluate(X_test, Y_test, show_accuracy=True)[1]))
    
def main():
    part4('part3_1.h5')
    part4('part3_2.h5')
if __name__ == '__main__':
    main()
