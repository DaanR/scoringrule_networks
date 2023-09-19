import keras
from keras.layers import Dense, Reshape
import tensorflow as tf

import sys

from examples.example_data_generation.generate_example_data import generate_bim_gaussian_univar_data
from multivariate_models.energy_score_ensembles.energy_score_loss import energy_score_loss
sys.path.append('..')  # Add the parent folder



def ES_loss_model(in_shape, n_pts, odim, layer_size, hidden_layers):
    model = keras.Sequential()
    model.add(keras.Input(shape=(in_shape)))
    for i in range(hidden_layers):
        model.add(Dense(layer_size, activation="relu", kernel_initializer = tf.keras.initializers.GlorotUniform()))
    model.add(Dense(n_pts * odim, kernel_initializer = tf.keras.initializers.GlorotUniform()))
    model.add(Reshape((n_pts, odim)))
    model.summary()
    model.compile(loss=energy_score_loss, optimizer="adam")
    return model


def do_experiment(model, x_train, y_train, x_test, y_test):
    ''' Training the model '''
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)

    ''' Making predictions '''
    y_pred = model.predict(x_test)
    print(y_pred.shape)
    #TODO: implement visualization code


if __name__ == "__main__":
    input_dim = 10
    target_dim = 2
    nr_pts = 150

    epochs = 15

    n_train = 10000
    n_test = 1000

    '''
        This first experiment is aimed at showing how Gaussian CRPS works with perfect data.
        We enter Gaussian distributed data into the model, and we find good distributions.
    '''
    print("Experiment 1: Gaussian target distributions, with Gaussian model")

    ''' Defining the model '''
    model = model = ES_loss_model(input_dim, nr_pts, target_dim, 64, 4)

    ''' Generating dummy data '''
    x_train, y_train = generate_bim_gaussian_univar_data(input_dim, target_dim, n_train)
    x_test, y_test = generate_bim_gaussian_univar_data(input_dim, target_dim, n_test)

    print("Data shapes")
    print("x_train", x_train.shape, "y_train", y_train.shape)
    print("x_test", x_test.shape, "y_test", y_test.shape)

    do_experiment(model, x_train, y_train, x_test, y_test)
