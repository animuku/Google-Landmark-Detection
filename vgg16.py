import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,BatchNormalization,ZeroPadding2D
from keras.layers.core import Dense,Flatten,Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=r"/home/anirudh/Desktop/Major Project/small_train",
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size


def VGG_16(num_classes, input_shape):
    model=Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu',input_shape=input_shape))
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D((2,2),strides=(2,2)))

    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D((2,2),strides=(2,2)))

    model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
    model.add(ZeroPadding2D(1,))
    model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(256,kernel_size=(1,1),activation='relu'))
    model.add(MaxPool2D((2,2),strides=(2,2)))

    model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(512,kernel_size=(1,1),activation='relu'))
    model.add(MaxPool2D((2,2),strides=(2,2)))

    model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(512,kernel_size=(1,1),activation='relu'))
    model.add(MaxPool2D((2,2),strides=(2,2)))


    model.add(Flatten())
    model.add(Dense((4096),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense((4096),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense((num_classes),activation='softmax'))

    return model

model=VGG_16(25,(128,128,3))

model.summary()



