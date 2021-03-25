#Creating a custom CNN for MNIST

from tensorflow import keras
import numpy as np


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

print(X_train_full.shape)
print(y_train_full.shape)
print(X_test.shape)
print(y_test.shape)

#Creating a validation set

X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

#Normalizing the data

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7

X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

#CNN Architecture 1

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(1,1), padding="same", input_shape=[28,28,1], activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])

#print(model.summary())

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=2)
test_loss = model.evaluate(X_test, y_test)
print(test_loss)



