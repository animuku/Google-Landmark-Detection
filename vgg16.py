import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,BatchNormalization,ZeroPadding2D
from keras.layers.core import Dense,Flatten,Dropout
from keras.objectives import SGD

def VGG_16(num_classes, input_shape):
    model=Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(2,2),stride=(2,2))

    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(2,2),stride=(2,2))

    model.add(Conv2D(256,kernel_size=(3,3),activation='relu')))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(256,kernel_size=(3,3),activation='relu')))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(256,kernel_size=(1,1),activation='relu')))
    model.add(MaxPool2D(2,2),stride=(2,2))

    model.add(Conv2D(512,kernel_size=(3,3),activation='relu')))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(512,kernel_size=(3,3),activation='relu')))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(512,kernel_size=(1,1),activation='relu')))
    model.add(MaxPool2D(2,2),stride=(2,2))

    model.add(Conv2D(512,kernel_size=(3,3),activation='relu')))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(512,kernel_size=(3,3),activation='relu')))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(512,kernel_size=(1,1),activation='relu')))
    model.add(MaxPool2D(2,2),stride=(2,2))


    model.add(Flatten())
    model.add(Dense(4096),activation='relu')
    model.add(Dropout(0.5))
    model.add(Dense(4096),activation='relu')
    model.add(Dropout(0.5))
    model.add(Dense(num_classes),activation='softmax')

    return model

    

