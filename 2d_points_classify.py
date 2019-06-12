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

# Convert training outputs into one-hot format.
# This might be useful if you plan to play with different
# NN structures, optimizers, activation and loss functions.
y_train_one_hot = np_utils.to_categorical(y_train)

input_layer = keras.Input(shape=(2,))
# hidden_layer_1 = layers.Dense(2, activation='relu')(input_layer)
output_layer = layers.Dense(1, activation='sigmoid')(input_layer)

model = keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['acc', 'mse'])

model.summary()

# Train
history = model.fit(x_train, y_train, epochs=10)

# Use
y_test = model.predict(x_test)

# Save the trained model to disk
model.save("2d_points_clas_model.h5")

# ======================================
# Plot training and test data
# ======================================

import matplotlib.pyplot as plt

# Figure 1
# --------
fig1 = plt.figure(1, figsize=(10,5)) # size in inches
fig1.suptitle('2d points classify')

# Remember that x0 is x coord., x1 is y coordinate,
#               y is color (somewhat z coordinate)
axs1 = fig1.add_subplot(1, 2, 1)
axs1.scatter(x_train[:,0], x_train[:,1], 1, y_train)
axs1.grid(True)
axs1.set_title('Training data')

axs2 = fig1.add_subplot(1, 2, 2)
axs2.scatter(x_test[:,0], x_test[:,1], 1, y_test[:,0])
axs2.grid(True)
axs2.set_title('Test data')

# Required to identify the necessary range on the horizontal axis of the plot
min_x0_test = np.amin(x_test[:,0])
max_x0_test = np.amax(x_test[:,0])
h_var = np.linspace(min_x0_test, max_x0_test)

weights = model.get_weights()[0]
biases = model.get_weights()[1]
###print(weights)
###print(biases)

a = -weights[0]/weights[1]
b = -biases[0]/weights[1]

plt.plot(h_var, [a*i + b for i in h_var], color='red')

# ======================================
# Plot training process graphs
# ======================================

# Figure 2
# --------
fig2 = plt.figure(2, figsize=(15,5)) # size in inches
fig2.suptitle('Training process')

axs3 = fig2.add_subplot(1, 3, 1)
axs3.plot(history.history['acc'])
axs3.grid(True)
axs3.set_title('acc')
axs3.set_ylim([0., 1.01])

axs4 = fig2.add_subplot(1, 3, 2)
axs4.plot(history.history['mse'])
axs4.grid(True)
axs4.set_title('mse')
axs4.set_ylim([0., None])

axs5 = fig2.add_subplot(1, 3, 3)
axs5.plot(history.history['loss'])
axs5.grid(True)
axs5.set_title('loss')
axs5.set_ylim([0., None])

# This should be called only once in the very end
plt.show()
