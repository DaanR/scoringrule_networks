import keras
from keras.layers import Dense, Reshape

from examples.example_data_generation.generate_example_data import generate_bim_gaussian_univar_data
from multivariate_models.conditional_crps.ccrps import conditional_CRPS_mixtures, postprocess_model_output
import sys

sys.path.append('..')  # Add the parent folder

from multivariate_models.conditional_crps.ccrps import partial_mixture_CCRPS

"""
     Defines a simple example model, with three hidden layers and "adam" loss.

     :param input_dim: an integer, denoting the size of the input layer
     :param output_dim: an integer, denoting the dimensionality of the target distribution
     :n_mixture: an integer, denoting the specified number of mixtures.

     :return: a keras model
"""


def mixture_ccrps_model(import_dim, target_dim, n_mixtures):
    # Determine which loss is going to be used. Use the other loss function as a metric

    model = keras.Sequential()
    model.add(keras.Input(shape=(import_dim)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(n_mixtures * (1 + 2 * target_dim + int(round(target_dim * (target_dim-1)/2)))))
    model.add(Reshape((n_mixtures, 1 + 2 * target_dim + int(round(target_dim * (target_dim-1)/2)))))
    model.summary()
    model.compile(loss=partial_mixture_CCRPS, optimizer="adam")
    return model


"""
    In this example, we define a simple ANN model, and train it via Gaussian CRPS or Log-score loss
    We will then visualize the predictions, and compute probability integral transformations
"""
if __name__ == "__main__":

    input_dim = 10
    target_dim = 5
    n_mixtures = 2

    epochs = 15

    n_train = 10000
    n_test = 1000

    ''' Defining the model '''
    model = mixture_ccrps_model(input_dim, target_dim, n_mixtures)

    ''' Generating dummy data '''
    x_train, y_train = generate_bim_gaussian_univar_data(input_dim, target_dim, n_train)
    x_test, y_test = generate_bim_gaussian_univar_data(input_dim, target_dim, n_test)

    print("Data shapes")
    print("x_train", x_train.shape, "y_train", y_train.shape)
    print("x_test", x_test.shape, "y_test", y_test.shape)

    ''' Training the model '''
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)

    ''' Making predictions '''
    y_pred = model.predict(x_test)

    # Convert the predictions to tensors of denoting the means, standard deviations and weights
    means, covs, weights = postprocess_model_output(y_pred, target_dim)

    print(means.shape)
    print(covs.shape)
    print(weights.shape)

    #TODO, make a visualization

