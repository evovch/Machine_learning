from csg_2d_data_set_generator import csg_2d_data_set_generator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# ==============================================================================

if __name__ == "__main__":

    n_features = 1000 # input
    n_classes = 4     # output
    n_batches = 10000 # batches per epoch
    batch_size = 1    # samples per batch
    n_epochs = 20

    genObj_train = csg_2d_data_set_generator(n_features, n_classes, n_batches, batch_size)

    input_layer = keras.Input(shape=(n_features,))
    output_layer = layers.Dense(n_classes, activation='sigmoid')(input_layer)
    optim = optimizers.SGD(lr=1.0)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optim,
                loss='categorical_crossentropy',
                metrics=['acc', 'mse'])

    model.fit_generator(generator=genObj_train,
#                        steps_per_epoch=n_batches, # optional; if not specified, generator.__len__() will be used
                        epochs=n_epochs,
#                        validation_data=genObj_test,
#                        validation_steps=n_batches # optional; if not specified, validation_data.__len__() will be used
                        workers=4,
                        use_multiprocessing=True)

    exit()
