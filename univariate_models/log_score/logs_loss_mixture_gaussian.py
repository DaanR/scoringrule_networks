import tensorflow as tf
from math import pi

# Allow for the parentfolder to be imported
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from mixture_gaussian_helpers import repeat_var, preprocess_mixture_output

"""
    Implements log score loss for a mixture Gaussian distribution.
    
    :param y_true: a (batch_size, n_targets) shaped tensor
    :param means: a (batch_size, n_mixtures, n) shaped tensor containing each mixture's mean
    :param stdevs: a (batch_size, n_mixtures, d) shaped tensor containing each mixture's standard deviation
    :param weights: a (batch_size, n_mixtures, n_targets) shaped tensor, containing each mixture's weight.
"""
def mixture_gaussian_logS(y_true, means, stdevs, weights, eps = 1e-37):
    n_mixtures = means.shape[1]
    
    # First transpose the last two dimensions, we might want to remove this transposing later on
    # shapes are now (batch_size, d, n_mixtures)
    means = tf.transpose(means, perm=[0, 2, 1])
    stdevs = tf.transpose(stdevs, perm=[0, 2, 1])
    weights = tf.transpose(weights, perm=[0, 2, 1])
    
    # Create a (batch_size, d, n_mixtures) shaped observation variable
    true2 = repeat_var(y_true, n_mixtures)
    
    # Implement the density manually. This could also have been done via tensorflow-probability
    dens = weights * tf.exp(-0.5 * (true2 - means)**2 / stdevs**2) / ((2 * pi)**0.5 * stdevs)
    
    # Sum to get the total density
    dens = tf.reduce_sum(dens, axis=2)
    
    # We set a minimum lower bound. This will prevent NaN's.
    # This is an elaborate way of creating an (bathc_size, n_targets) shaped constant tensor
    # As of now, I haven't found any other way that doesn't crash during model compiling (tf uses None sized dimensions)
    lowerBound = 0. * dens + eps
    
    return -tf.reduce_mean(tf.math.log(tf.maximum(lowerBound, dens)))



"""
    Wrapper to use mixture Gaussian Log Score as a keras loss function
    
    :param y_true: an (batch_size, n_targets) shaped tensor
    :param y_pred: an (batch_size, n_mixtures, 3*n_targets) shaped tensor
    
    :return the mean mixture Gaussian CRPS
"""
def mixture_gaussian_logS_loss(y_true, y_pred):
    n_targets = y_true.shape[1]
    means, stdevs, weights = preprocess_mixture_output(y_pred, n_targets)

    return mixture_gaussian_logS(y_true, means, stdevs, weights)