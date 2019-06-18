import numpy as np
import random as rnd

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from cloud_of_points import draw_2d_cloud_batch

from csg_2d_shape import get_primitive_names

from csg_2d_square import csg_2d_square                       # 0
from csg_2d_rectangle import csg_2d_rectangle                 # 1
from csg_2d_trapezoid import csg_2d_trapezoid                 # 2
from csg_2d_generic_trapezoid import csg_2d_generic_trapezoid # 3
from csg_2d_para import csg_2d_para                           # 4
from csg_2d_full_circle import csg_2d_full_circle             # 5
from csg_2d_full_ellipse import csg_2d_full_ellipse           # 6

# ==============================================================================
# GENERATOR
# ==============================================================================

class csg_2d_data_set_generator(keras.utils.Sequence):

    def __init__(self, n_features, n_classes, n_batches, batch_size):
        self.n_features = n_features # input
        self.n_classes = n_classes   # output
        self.n_batches = n_batches   # batches per epoch
        self.batch_size = batch_size # samples per batch

    def __getitem__(self, index):
        x, y, f = self.__data_generation()
        return x, y, f

    def __len__(self):
        return self.n_batches

    # Output shape: (batch_size, n_features),
    # where n_features/2 is the number of points in the cloud
    # [[xy xy ... xy][xy xy ... xy]...[xy xy ... xy]]
    def __data_generation(self):
        x = np.zeros([self.batch_size, self.n_features]) # input
        y = np.zeros([self.batch_size, self.n_classes])  # output
        f = np.zeros(self.batch_size, dtype = int)
        for i in range(self.batch_size):
            f[i] = rnd.choice(range(self.n_classes))
            if (f[i] == 0):
                prim = csg_2d_square()
                prim.gen_random()
                x[i] = np.reshape(prim.gen_cloud_on_boundary(int(self.n_features/2)), self.n_features)
            elif (f[i] == 1):
                prim = csg_2d_rectangle()
                prim.gen_random()
                x[i] = np.reshape(prim.gen_cloud_on_boundary(int(self.n_features/2)), self.n_features)
            elif (f[i] == 2):
                prim = csg_2d_trapezoid()
                prim.gen_random()
                x[i] = np.reshape(prim.gen_cloud_on_boundary(int(self.n_features/2)), self.n_features)
            elif (f[i] == 3):
                prim = csg_2d_generic_trapezoid()
                prim.gen_random()
                x[i] = np.reshape(prim.gen_cloud_on_boundary(int(self.n_features/2)), self.n_features)
            elif (f[i] == 4):
                prim = csg_2d_para()
                prim.gen_random()
                x[i] = np.reshape(prim.gen_cloud_on_boundary(int(self.n_features/2)), self.n_features)
            elif (f[i] == 5):
                prim = csg_2d_full_circle()
                prim.gen_random()
                x[i] = np.reshape(prim.gen_cloud_on_boundary(int(self.n_features/2)), self.n_features)
            elif (f[i] == 6):
                prim = csg_2d_full_ellipse()
                prim.gen_random()
                x[i] = np.reshape(prim.gen_cloud_on_boundary(int(self.n_features/2)), self.n_features)
#            elif (f[i] == 7): # cirlce
#            elif (f[i] == 8): # ellipse
            y[i] = keras.utils.to_categorical(f[i], num_classes=self.n_classes)
        return x, y, f

# ==============================================================================

if __name__ == "__main__":

    n_features = 500  # input
    n_classes = 7     # output
    n_batches = 1     # batches per epoch
    batch_size = 20   # samples per batch

    genObj_train = csg_2d_data_set_generator(n_features, n_classes, n_batches, batch_size)
    x_train, y_train, f = genObj_train.__getitem__(0)

    draw_2d_cloud_batch(x_train, get_primitive_names(f))

    plt.show()

    exit()
