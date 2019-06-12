# ======================================
# x is input (2D)
# x0 is x coordinate
# x1 is y coordinate
# y is output (1D)
# f is input object class (color)
# ======================================

# ======================================
# Prepare training and test data
# ======================================

import random as rnd
import numpy as np

n_output_classes = 2 # number of clusters
xc = [0.25, 0.25, 0.75, 0.75] # centers of the clusters
sigma = 0.05 # width of the clusters

# training data
n_train = 10000
x_train = np.zeros([n_train,2])
y_train = np.zeros([n_train])
for i in range(n_train):
    f = rnd.choice([0,1])
    x_train[i] = [rnd.gauss(0., sigma) + xc[f*2+0],
                  rnd.gauss(0., sigma) + xc[f*2+1]]
    y_train[i] = f

# test data
n_test = 10000
x_test = np.zeros([n_test,2])
###y_test = np.zeros([n_test])
for i in range(n_test):
    f = rnd.choice([0,1])
    x_test[i] = [rnd.gauss(0., sigma) + xc[f*2+0],
                 rnd.gauss(0., sigma) + xc[f*2+1]]

# ======================================
# Build, train, and use the classifier
# ======================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils

# Convert training outputs into one-hot format
y_train_one_hot = np_utils.to_categorical(y_train)

input_layer = keras.Input(shape=(2,))
# hidden_layer_1 = layers.Dense(2, activation='relu')(input_layer)
output_layer = layers.Dense(1, activation='sigmoid')(input_layer)

model = keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# train
model.fit(x_train, y_train, epochs=10)

# use
y_test = model.predict(x_test)

# ======================================
# Plot training and testdata
# ======================================

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)
axs[0].scatter(x_train[:,0], x_train[:,1], 1, y_train) # remember, x0 is x coord., x1 is y ccordinate, y is color (somewhat z coordinate)
axs[0].grid(True)
axs[1].scatter(x_test[:,0], x_test[:,1], 1, y_test[:,0])
axs[1].grid(True)

# Required to identify the necessary range on the horizontal axis of the plot
min_x0_test = np.amin(x_test[:,0])
max_x0_test = np.amax(x_test[:,0])
h_var = np.linspace(min_x0_test, max_x0_test)

weights = model.get_weights()[0]
biases = model.get_weights()[1]
print(weights)
print(biases)

a = -weights[0]/weights[1]
b = -biases[0]/weights[1]

plt.plot(h_var, [a*i + b for i in h_var], color='red')

plt.show()
