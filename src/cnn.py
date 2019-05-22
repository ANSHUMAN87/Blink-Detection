from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, AvgPool2D, Flatten
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense

#make the convolution neural network
def blink_detector_old(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding = 'same', name='image_array', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (2,2), padding= 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2,2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    '''model.add(Dense(2))
    model.add(Activation("softmax", name='predictions'))'''
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model

if __name__ == "__main__":
    model = blink_detector((100, 100, 3))
    model.summary()
