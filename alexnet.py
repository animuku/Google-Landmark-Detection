import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPool2D, ZeroPadding2D, BatchNormalization,Activation
import cv2
import numpy as numpy


def AlexNet(input_shape,n_classes):
    model=Sequential()
    model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4), input_shape=input_shape))
    model.add(ZeroPadding2D(2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=3,strides=2))

    model.add(Conv2D(filters=256,kernel_size=(5,5)))
    model.add(ZeroPadding2D(2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=3,strides=2))

    model.add(Conv2D(filters=384,kernel_size=(3,3)))
    model.add(ZeroPadding2D(1))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=384,kernel_size=(3,3)))
    model.add(ZeroPadding2D(1))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256,kernel_size=(3,3)))
    model.add(ZeroPadding2D(1))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model

model=AlexNet(20)
model.summary()





    