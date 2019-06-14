# ======================================
# The goal is to train a neural network
# which identifies horizonal lines out 
# of a bunch of random lines
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

# ======================================
# Plot training and test data
# ======================================
def DrawData(fig_idx, p_x, p_y, title):

    # Calculate angles and lengths
    n = len(p_x)
    radToDeg = 180. / np.pi
    alphas = np.zeros([n])
    lengths = np.zeros([n])
    for i in range(n):
        dx0 = p_x[i][2] - p_x[i][0] # aka dx coordinate
        dx1 = p_x[i][3] - p_x[i][1] # aka dy coordinate
        alphas[i] = radToDeg * np.arctan(dx1/dx0) # line inclination angle
        lengths[i] = np.sqrt(dx0*dx0+dx1*dx1)

    # Separate true from false
    alpha_true = alphas[p_y[:,0]==1]
    alpha_false = alphas[p_y[:,0]==0]
    lengths_true = lengths[p_y[:,0]==1]
    lengths_false = lengths[p_y[:,0]==0]

    # Plot
    fig = plt.figure(fig_idx, figsize=(10,5)) # size in inches
    fig.suptitle(title)

    # Used range is [-90;90]
    axs1 = fig.add_subplot(1, 2, 1)
    ###axs1.hist(alphas, bins=100, range=[-100., 100.])
    axs1.scatter(alpha_true, lengths_true, 1, 'red')
    axs1.grid(True)
    axs1.set_title('True')
    axs1.set_xlim([-100., 100.])
    axs1.set_ylim([-0.5, 2.0])

    axs2 = fig.add_subplot(1, 2, 2)
    ###axs2.hist(lengths, bins=20)
    axs2.scatter(alpha_false, lengths_false, 1, 'blue')
    axs2.grid(True)
    axs2.set_title('False')
    axs2.set_xlim([-100., 100.])
    axs2.set_ylim([-0.5, 2.0])

# ======================================
# Plot individual training data samples
# ======================================
def DrawIndividualSamplesLine(fig_idx, p_x, p_y):

    limit = 500

    fig = plt.figure(fig_idx, figsize=(10,5)) # size in inches
    fig.suptitle('Data samples')

    axsT = fig.add_subplot(1, 2, 1)
    axsF = fig.add_subplot(1, 2, 2)

    counterTrue = 0
    counterFalse = 0

    for i in range(len(p_y)):
        edge_p = np.linspace(0., 1., 2)
        if (p_y[i][0] == 1):
            if (counterTrue < limit):
                axsT.plot(p_x[i][0] + (p_x[i][2]-p_x[i][0])*edge_p,
                          p_x[i][1] + (p_x[i][3]-p_x[i][1])*edge_p)
                counterTrue+=1

        else:
            if (counterFalse < limit):
                axsF.plot(p_x[i][0] + (p_x[i][2]-p_x[i][0])*edge_p,
                          p_x[i][1] + (p_x[i][3]-p_x[i][1])*edge_p)
                counterFalse+=1


    axsT.grid(True)
    axsT.set_title('true')
    axsF.grid(True)
    axsF.set_title('false')

# ==============================================================================

def GenDataSetLine(n):
    x = np.zeros([n, 4])
    y = np.zeros([n, 1])
    for i in range(n):
        f = rnd.choice([0,1])
        if (f==0): # not horizontal line
            x[i] = [rnd.uniform(0., 1.), rnd.uniform(0., 1.),
                    rnd.uniform(0., 1.), rnd.uniform(0., 1.)]
        else: # horizontal line
            tmp = rnd.uniform(0., 1.)
            x[i] = [rnd.uniform(0., 1.), tmp,
                    rnd.uniform(0., 1.), tmp]
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

###n_output_classes = 2 # two possible outcomes - horizontal line / not horizontal line

# training data
n_train = 100000
x_train, y_train = GenDataSetLine(n_train)

# test data
n_test = 10000
x_test, y_test = GenDataSetLine(n_test)

# ==============================================================================

# ======================================
# Build, train, and use the classifier
# ======================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
#from keras.utils import np_utils

input_layer = keras.Input(shape=(4,))
hidden_layer_1 = layers.Dense(2, activation='tanh')(input_layer)
output_layer = layers.Dense(1, activation='sigmoid')(hidden_layer_1)

optim = optimizers.SGD(lr=1.0)

model = keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=optim,
              loss='binary_crossentropy',
              metrics=['acc', 'mse'])

if (LoadModel==True):
    # Load the trained model from disk
    model.load_weights("2d_lines_clas_model.h5")
else:
    # Train
    history = model.fit(x_train,
                        y_train,
    #                    batch_size=100,
                        epochs=20)
    # Save the trained model to disk
    model.save_weights("2d_lines_clas_model.h5")

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
DrawData(3, x_test, y_test, 'Test data')
DrawData(4, x_test, a_test_pro, 'Test data prediction')
DrawIndividualSamplesLine(5, x_train, y_train)
DrawIndividualSamplesLine(6, x_test, a_test_pro)

# ===============================================
# This should be called only once in the very end
plt.show()
exit()
