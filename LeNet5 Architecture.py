#Setup
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#Data Loading
mnist = tf.keras.datasets.mnist
(X_train,y_train),(X_test, y_test) = mnist.load_data()

#SPecifying the number of rows and columns
rows,cols = 28,28

#reshaping the inputs
X_train = X_train.reshape(X_train.shape[0], rows,cols,1)
X_test = X_test.reshape(X_test.shape[0], rows,cols,1)
input_shape = (rows,cols,1)

#normalizing the inputs
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test/255.0

#One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

def build_lenet(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters = 6, kernel_size=(5,5),strides=(1,1), input_shape=(28,28,1), activation='tanh'),
        tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), activation='tanh'),
        tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=120, activation='tanh'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=84, activation='tanh'),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


lenet = build_lenet(input_shape)

#specifying the number of epochs
epochs = 10

#Transformation / Reshape into 28x28 pixel

X_train = X_train.reshape(X_train.shape[0], 28,28)
print("Training Data", X_train.shape, y_train.shape)

X_test = X_test.reshape(X_test.shape[0], 28,28)
print("Testing Data", X_test.shape, y_test.shape)

#Fitting the model

history = lenet.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=1)
loss,acc = lenet.evaluate(X_test, y_test)
print(lenet.summary())
print('Accuracy:', acc)

#Predicting an Image
image_index=4444
plt.imshow(X_test[image_index].reshape(28,28), cmap="Greys")
pred = lenet.predict(X_test[image_index].reshape(1, rows, cols,1))
print(pred.argmax())