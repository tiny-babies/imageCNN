#  implement a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset.

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.callbacks import EarlyStopping
from tensorflow import constant
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

import tensorflow as tf

# Solve SSL Certificate problem
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# training data: 5000 images that are 32x32 pixels and RGB
assert x_train.shape == (50000, 32, 32, 3)

# testing data: 1000 images that are 32x32 pixels and RGB
assert x_test.shape == (10000, 32, 32, 3)

# Training labels of integers 0-9 for 5000 samples
assert y_train.shape == (50000, 1)

# Testing labels of integers 0-9 for 5000 samples
assert y_test.shape == (10000, 1)


# simple deep net with the following structure:

# Input layer: Specify the appropriate shape (according to CIFAR-10 image shape)
# Convolutional layer: 7x7 kernel, 64 output channels, ReLU activation
# Max pooling layer: 2x2 pool size, stride 2
# Convolutional layer: 3x3 kernel, 128 output channels, ReLU activation
# Max pooling layer: 2x2 pool size, stride 2
# Convolutional layer: 3x3 kernel, 256 output channels, ReLU activation
# Max pooling layer: 2x2 pool size, stride 2
# Flatten the output
# Dropout layer (0.5 probability)
# Dense layer with softmax activation

# Using docs from https://keras.io/guides/sequential_model/
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer

model = Sequential()
model.add(InputLayer(input_shape=(32,32,3)))
model.add(Conv2D(64, (7, 7), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()




# https://keras.io/api/callbacks/early_stopping/
callback = EarlyStopping(monitor='loss', patience=3)

# https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class
# Using the categorical crossentropy loss function because it is appropriate for
# our classification problem with multiple classes, and it provides one_hot representation
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Must one_hot the training data

y_train_one_hot = to_categorical(y_train, 10)
y_train_one_hot = constant(y_train_one_hot, shape=[50000, 10])

# Train the model on the training data, using the last 20% of the data for validation.

history = model.fit(x_train, y_train_one_hot, batch_size=256,
                    epochs=17, validation_split=0.2)


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()