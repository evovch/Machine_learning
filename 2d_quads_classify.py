# ======================================
# The goal is to train a neural network
# which identifies rectangles out 
# of a bunch of random quad patches
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

    n = len(p_x) # number of quads
    cos_alpha_distr = np.zeros([4*n])
    for i in range(n):
        for j in range(4): # j - current
            k = 3 if (j==0) else j-1 # k - previous
            q = 0 if (j==3) else j+1 # q - next

            pc = np.array([p_x[i][j*2+0], p_x[i][j*2+1]]) # current
            pp = np.array([p_x[i][k*2+0], p_x[i][k*2+1]]) # previous
            pn = np.array([p_x[i][q*2+0], p_x[i][q*2+1]]) # next

            d0 = pp-pc # vector from current to previous
            d1 = pn-pc # vector from current to next
            cos_alpha = np.dot(d0, d1) / (np.linalg.norm(d0)*np.linalg.norm(d1))
            cos_alpha_distr[4*i+j] = cos_alpha
    
    fig = plt.figure(fig_idx, figsize=(10,10)) # size in inches
    fig.suptitle(title)

    axs1 = fig.add_subplot()
    axs1.hist(cos_alpha_distr, bins=100)
    axs1.grid(True)

# ======================================
# Plot individual training data samples
# ======================================
def DrawIndividualSamplesQuad(fig_idx, p_x, p_y):

    limit = 10

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
                axsT.plot(p_x[i][2] + (p_x[i][4]-p_x[i][2])*edge_p,
                          p_x[i][3] + (p_x[i][5]-p_x[i][3])*edge_p)
                axsT.plot(p_x[i][4] + (p_x[i][6]-p_x[i][4])*edge_p,
                          p_x[i][5] + (p_x[i][7]-p_x[i][5])*edge_p)
                axsT.plot(p_x[i][6] + (p_x[i][0]-p_x[i][6])*edge_p,
                          p_x[i][7] + (p_x[i][1]-p_x[i][7])*edge_p)
                counterTrue+=1
        else:
            if (counterFalse < limit):
                axsF.plot(p_x[i][0] + (p_x[i][2]-p_x[i][0])*edge_p,
                          p_x[i][1] + (p_x[i][3]-p_x[i][1])*edge_p)
                axsF.plot(p_x[i][2] + (p_x[i][4]-p_x[i][2])*edge_p,
                          p_x[i][3] + (p_x[i][5]-p_x[i][3])*edge_p)
                axsF.plot(p_x[i][4] + (p_x[i][6]-p_x[i][4])*edge_p,
                          p_x[i][5] + (p_x[i][7]-p_x[i][5])*edge_p)
                axsF.plot(p_x[i][6] + (p_x[i][0]-p_x[i][6])*edge_p,
                          p_x[i][7] + (p_x[i][1]-p_x[i][7])*edge_p)
                counterFalse+=1

    axsT.grid(True)
    axsT.set_title('true')
    axsF.grid(True)
    axsF.set_title('false')

# ==============================================================================

def GenDataSetQuad(n):
    x = np.zeros([n, 8])
    y = np.zeros([n, 1])
    tmp = np.zeros([8])
    for i in range(n):
        f = rnd.choice([0,1])
        if (f==0): # random quad
            x[i] = [rnd.uniform(0., 1.), rnd.uniform(0., 1.),
                    rnd.uniform(0., 1.), rnd.uniform(0., 1.),
                    rnd.uniform(0., 1.), rnd.uniform(0., 1.),
                    rnd.uniform(0., 1.), rnd.uniform(0., 1.)]
        else: # rectangle
            tmp = [rnd.uniform(0., 1.), rnd.uniform(0., 1.),
                    rnd.uniform(0., 1.), rnd.uniform(0., 1.),
                    rnd.uniform(0., 1.), rnd.uniform(0., 1.),
                    rnd.uniform(0., 1.), rnd.uniform(0., 1.)]

            x[i][0] = tmp[0]
            x[i][1] = tmp[1]
            x[i][2] = tmp[2]
            x[i][3] = tmp[3]

            p0x = tmp[0]
            p0y = tmp[1]
            p1x = tmp[2]
            p1y = tmp[3]
            p2x = tmp[4]
            p2y = tmp[5]

            bx = -(p1y-p0y)
            by = p1x-p0x
            ax = p2x-p1x
            ay = p2y-p1y

            ab = ax*bx + ay*by
            b_mag2 = bx*bx+by*by

            a1x = ab*bx/b_mag2
            a1y = ab*by/b_mag2

            x[i][4] = x[i][2] + a1x
            x[i][5] = x[i][3] + a1y
            x[i][6] = x[i][0] + a1x
            x[i][7] = x[i][1] + a1y

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

###n_output_classes = 2 # two possible outcomes - rectangle / random quad patch

# training data
n_train = 100000
x_train, y_train = GenDataSetQuad(n_train)

# test data
n_test = 10000
x_test, y_test = GenDataSetQuad(n_test)

# ==============================================================================

# ======================================
# Build, train, and use the classifier
# ======================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.utils import np_utils

input_layer = keras.Input(shape=(8,))
hidden_layer_1 = layers.Dense(2, activation='tanh')(input_layer)
output_layer = layers.Dense(1, activation='sigmoid')(hidden_layer_1)

optim = optimizers.SGD(lr=1.0)

model = keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=optim,
              loss='binary_crossentropy',
              metrics=['acc', 'mse'])

if (LoadModel==True):
    # Load the trained model from disk
    model.load_weights("2d_quads_clas_model.h5")
else:
    # Train
    history = model.fit(x_train,
                        y_train,
    #                    batch_size=100,
                        epochs=20)
    # Save the trained model to disk
    model.save("2d_quads_clas_model.h5")

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
DrawIndividualSamplesQuad(5, x_train, y_train)
DrawIndividualSamplesQuad(6, x_test, a_test_pro)

# ===============================================
# This should be called only once in the very end
plt.show()
exit()
