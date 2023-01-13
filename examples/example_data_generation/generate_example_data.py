import numpy as np
from numpy.random import normal, randint

"""
    Generates a dummy dataset with Gaussian distributed target variables. Entries in x and y are related (sampled from an identical distribution)
    but entries in y are warped by linear transformed).
    
    Models will attempt to learn the (linearly transformed versions of the) input distribution from the inputs x
    mapped those mappings.
    
    :param input_dim: an integer, denoting the size of the input layer
    :param output_dim: an integer, denoting the dimensionality of the target distribution
    :param n: the required number of datapoints
     
    :return x: an (n, input_dim) shaped numpy array, of input data.
    :return y: an (n, output_dim) shaped numpy array, of target data
"""
def generate_gaussian_univar_data(input_dim, target_dim, n):
    x = np.empty((n, input_dim))
    y = np.empty((n, target_dim))
    for idx, (x_val, y_val) in enumerate(zip(x, y)):
        random_mean = normal(loc = 0, scale = 1)
        random_stdev = np.abs(normal(loc = 1, scale = 1)) + 1e-5 # Small addition to ensure strict positivity
        samples = normal(loc = random_mean, scale = random_stdev, size = input_dim + target_dim)
        this_x = samples[:input_dim]
        this_y = [(i+1) * s - i for i, s in enumerate(samples[input_dim:])]
        x[idx,:] = this_x
        y[idx,:] = this_y
    return x, y



"""
    Generates a dummy dataset with bimodal Gaussian distributed target variables. Entries in x and y are related (sampled from an identical distribution)
    but entries in y are warped by linear transformed).
    
    Models will attempt to learn the (linearly transformed versions of the) input distribution from the inputs x
    mapped those mappings.
    
     :param input_dim: an integer, denoting the size of the input layer
     :param output_dim: an integer, denoting the dimensionality of the target distribution
     :param n: the required number of datapoints
     
     :return x: an (n, input_dim) shaped numpy array, of input data.
     :return y: an (n, output_dim) shaped numpy array, of target data
"""
def generate_bim_gaussian_univar_data(input_dim, target_dim, n):
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
