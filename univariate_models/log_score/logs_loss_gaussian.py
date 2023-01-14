import tensorflow as tf
from math import pi


# Allow for the parentfolder to be imported
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from gaussian_helpers import get_mu_sigma

"""
    Implementation of a keras custom loss function. It evaluates the Log-score over a Gaussian distribution/observation
    
    :param y_true: a (batch_size, n_targets) shaped tensor, containing the measurements
    :param y_pred: a (batch_size, 2*n_targets) shaped tensor, with:
                        y_pred[:,:n_targets] containing the forecasted means
                        y_pred[:,n_targets:] containing the forecasted standard deviations (will be softmax activated)
    :return: a (1,) shaped tensor, the mean CRPS computed over the forecasted distributions
"""
def gaussian_logS_loss(y_true, y_pred, eps = 1e-37):
    n_targets = y_true.shape[1]
    mu, sigma = get_mu_sigma(y_pred, n_targets)
    
    # Implement the density manually. This could also have been done via tensorflow-probability
    dens = tf.exp(-0.5 * (y_true - mu)**2 / sigma**2) / ((2 * pi)**0.5 * sigma)
    
    # We set a minimum lower bound. This will prevent NaN's.
    # This is an elaborate way of creating an (bathc_size, n_targets) shaped constant tensor
    # As of now, I haven't found any other way that doesn't crash during model compiling (tf uses None sized dimensions)
    lowerBound = 0. * dens + eps
    
    return -tf.reduce_mean(tf.math.log(tf.maximum(lowerBound, dens)))
    
if __name__ == "__main__":
    print(gaussian_logS_loss(tf.constant([[1.]]), tf.constant([[3,2.]])))