# ======================================
# x is input (2D)
# x0 is x coordinate
# x1 is y coordinate
# y is output (1D)
# f is input object class (color)
# a is prediction (activation)
# ======================================

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

LoadModel = False

# ==============================================================================

# ======================================
# Plot training process graphs
# ======================================
def DrawTrainingHistory(fig_idx, p_history):

    fig = plt.figure(fig_idx, figsize=(15,5)) # size in inches
    fig.suptitle('Training history')

    axs1 = fig.add_subplot(1, 3, 1)
    axs1.plot(history.history['acc'])
    axs1.grid(True)
    axs1.set_title('acc')
    axs1.set_ylim([0., 1.01])

    axs2 = fig.add_subplot(1, 3, 2)
    axs2.plot(history.history['mse'])
    axs2.grid(True)
    axs2.set_title('mse')
    axs2.set_ylim([0., None])

    axs3 = fig.add_subplot(1, 3, 3)
    axs3.plot(history.history['loss'])
    axs3.grid(True)
    axs3.set_title('loss')
    axs3.set_ylim([0., None])

def DrawData(fig_idx, p_x, p_y, title):

    # Separate true from false
    x_true = p_x[p_y[:,0]==1]
    x_false = p_x[p_y[:,0]==0]

    # Plot
    fig = plt.figure(fig_idx, figsize=(10,5)) # size in inches
    fig.suptitle(title)

    # Remember that x0 is x coord., x1 is y coordinate,
    #               y is color (somewhat z coordinate)
    axs1 = fig.add_subplot(1, 2, 1)
    axs1.scatter(x_true[:,0], x_true[:,1], 1, 'red')
    axs1.grid(True)
    axs1.set_title('True')
    axs1.set_xlim([-0.5, 1.5])
    axs1.set_ylim([-0.5, 1.5])

    axs2 = fig.add_subplot(1, 2, 2)
    axs2.scatter(x_false[:,0], x_false[:,1], 1, 'blue')
    axs2.grid(True)
    axs2.set_title('False')
    axs2.set_xlim([-0.5, 1.5])
    axs2.set_ylim([-0.5, 1.5])

# ======================================
# Plot training and test data
# ======================================
def DrawData2(fig_idx, x_train, y_train, x_test, y_test, title):

    fig = plt.figure(fig_idx, figsize=(10,5)) # size in inches
    fig.suptitle(title)

    # Remember that x0 is x coord., x1 is y coordinate,
    #               y is color (somewhat z coordinate)
    axs1 = fig.add_subplot(1, 2, 1)
    axs1.scatter(x_train[:,0], x_train[:,1], 1, y_train[:,0])
    axs1.grid(True)
    axs1.set_title('Training data')

    axs2 = fig.add_subplot(1, 2, 2)
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

# ==============================================================================

def GenDataSetPoint(n, xc, sigma):
    x = np.zeros([n_train, 2])
    y = np.zeros([n_train, 1])
    for i in range(n_train):
        f = rnd.choice([0,1])
        x[i] = [rnd.gauss(0., sigma) + xc[f*2+0],
                rnd.gauss(0., sigma) + xc[f*2+1]]
        y[i] = [f]
    return x, y

def PostProcess(a_test, threshold):
    a_test_pro = np.zeros(a_test.shape)
    for i in range(len(a_test)):
        for j in range(len(a_test[i])):
            if (a_test[i][j] < threshold):
                a_test_pro[i][j] = 0
            else:
                a_test_pro[i][j] = 1
            #if (a_test_pro[i][j] != y_test[i][j]):
            #    print('[', i, ',', j, ']: ', a_test[i][j], '(', y_test[i][j], ')')
    return a_test_pro

# ==============================================================================

# ======================================
# Prepare training and test data
# ======================================

###n_output_classes = 2 # number of clusters
xc = [0.25, 0.25, 0.75, 0.75] # centers of the clusters
sigma = 0.1 # width of the clusters

# training data
n_train = 1000000
x_train, y_train = GenDataSetPoint(n_train, xc, sigma)

# test data
n_test = 10000000
x_test, y_test = GenDataSetPoint(n_test, xc, sigma)

# ==============================================================================

# ======================================
# Build, train, and use the classifier
# ======================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.utils import np_utils

input_layer = keras.Input(shape=(2,))
# hidden_layer_1 = layers.Dense(2, activation='relu')(input_layer)
output_layer = layers.Dense(1, activation='sigmoid')(input_layer)

optim = optimizers.SGD(lr=1.0)

model = keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=optim,
              loss='binary_crossentropy',
              metrics=['acc', 'mse'])

if (LoadModel==True):
    # Load the trained model from disk
    model.load_weights("2d_points_clas_model.h5")
else:
    # Train
    history = model.fit(x_train,
                        y_train,
                        batch_size=100,
                        epochs=5)
    # Save the trained model to disk
    model.save("2d_points_clas_model.h5")

    DrawTrainingHistory(1, history)

model.summary()

# Use
###a_test = np.zeros([n_test, 1])
a_test = model.predict(x_test)

# ======================================
# PostProcessing
# ======================================

threshold = 0.5
a_test_pro = PostProcess(a_test, threshold)
MSE = np.square(np.subtract(a_test_pro, y_test)).mean()
print('MSE on the test data: ', MSE)

# ======================================
# Draw results
# ======================================

DrawData(2, x_train, y_train, 'Train data')
DrawData(3, x_test, a_test_pro, 'Test data')

# ===============================================
# This should be called only once in the very end
plt.show()
exit()
