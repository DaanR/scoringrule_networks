import keras
from keras.layers import Dense, Reshape
import numpy as np
from numpy.random import normal, randint

import sys
sys.path.append('..') # Add the parent folder

from univariate_models.crps.crps_loss_mixture_gaussian import mixture_gaussian_CRPS_loss, preprocess_mixture_output
from visualizations.visualize_univariate_mixture_gaussians import visualize_gaussian_predictions, do_gaussian_PIT


"""
     Defines a simple example model, with three hidden layers and "adam" loss.
     
     :param input_dim: an integer, denoting the size of the input layer
     :param output_dim: an integer, denoting the dimensionality of the target distribution
     :n_mixture: an integer, denoting the specified number of mixtures.
     
     :return: a keras model
"""
def gaussian_crps_model(import_dim, target_dim, n_mixtures):
    model = keras.Sequential()
    model.add(keras.Input(shape=(import_dim)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(3 * n_mixtures * target_dim))
    model.add(Reshape((n_mixtures, 3 * target_dim)))
    model.summary()
    model.compile(loss = mixture_gaussian_CRPS_loss, optimizer="adam")
    return model


"""
    Generates a dummy dataset. Entries in x and y are related (sampled from an identical distribution)
    but entries in y are warped by linear transformed).
    
    Models will attempt to learn the (linearly transformed versions of the) input distribution from the inputs x
    mapped those mappings.
    
     :param input_dim: an integer, denoting the size of the input layer
     :param output_dim: an integer, denoting the dimensionality of the target distribution
     :param n: the required number of datapoints
     
     :return x: an (n, input_dim) shaped numpy array, of input data.
     :return y: an (n, output_dim) shaped numpy array, of target data
"""
def generate_dummy_data(input_dim, target_dim, n):
    x = np.empty((n, input_dim))
    y = np.empty((n, target_dim))
    for idx, (x_val, y_val) in enumerate(zip(x, y)):
        
        # First mixture
        random_mean1 = normal(loc = 0, scale = 1)
        random_stdev1 = np.abs(normal(loc = 1, scale = 1)) + 1e-5 # Small addition to ensure strict positivity
        
        # Second mixture
        random_mean2 = normal(loc = 10, scale = 1)
        random_stdev2 = np.abs(normal(loc = 1, scale = 1)) + 1e-5 # Small addition to ensure strict positivity
        
        samples = normal(loc = random_mean1, scale = random_stdev1, size = input_dim + target_dim)
        samples2 = normal(loc = random_mean2, scale = random_stdev2, size = input_dim + target_dim)
        
        # 0.5 probability for each mixture
        randints = randint(0, 2, size = input_dim + target_dim)
        
        # This is in essence a mixture Gaussian with 50% probability for each mixture
        samples = randints * samples + (1 - randints) * samples2
        
        this_x = samples[:input_dim]
        this_y = [(i+1) * s - i for i, s in enumerate(samples[input_dim:])]
        x[idx,:] = this_x
        y[idx,:] = this_y
    return x, y




"""
    In this example, we define a simple ANN model, and train it via Gaussian CRPS loss.
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
    model = gaussian_crps_model(input_dim, target_dim, n_mixtures)
    
    ''' Generating dummy data '''
    x_train, y_train = generate_dummy_data(input_dim, target_dim, n_train)
    x_test, y_test = generate_dummy_data(input_dim, target_dim, n_test)

    ''' Training the model '''
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs)
    
    ''' Making predictions '''
    y_pred = model.predict(x_test)
      
    # Convert the predictions to tensors of denoting the means, standard deviations and weights
    mus, sigmas, weights = preprocess_mixture_output(y_pred, target_dim)
    
    # Convert the tensors to numpy arrays, which is easier for visualization
    mus = mus.numpy()
    sigmas = sigmas.numpy()
    weights = weights.numpy()
    
    ''' Visualizing the output '''
    examples = 5
    visualize_gaussian_predictions(y_test[:examples], mus[:examples], sigmas[:examples], weights[:examples])
#    
    ''' Probability integral transformations'''
    do_gaussian_PIT(y_test, mus, sigmas, weights)