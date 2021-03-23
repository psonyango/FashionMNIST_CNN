#SETUP

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

#Specifying input image shape

image_shape = (227,227,3)

#instantiating an empty model

np.random.seed(1000)

#Ceating the AlexNet DNN

model = Sequential([
    Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), input_shape=(image_shape), activation="relu",
           padding='valid'),
    MaxPooling2D(pool_size=(3, 3), padding='valid', strides=(2, 2)),
    Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation="relu"),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation="relu"),
    Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation="relu"),
    Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation="relu"),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Flatten(),
    Dense(units=4096, activation="relu"),
    Dropout(rate=0.5),
    Dense(units=4096, activation="relu"),
    Dropout(rate=0.5),
    Dense(units=1000, activation="softmax"),

])

print(model.summary())


