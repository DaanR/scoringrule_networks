import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow.math as tfm
import keras
from math import pi

"""
Activations mainly aimed to prevent negative (or zero) standard devations.
"""
def activations_stdev(stdevs):
    return 0.0001+keras.activations.softplus(stdevs)


"""
Splits the y_pred forecast into activated means and standard deviations

:param y_true a (batch_size, 2*n) shaped tensor.
:param n: an integer
:return mu: a (batch_size, n) shaped tensor, containing the forecasted means.
:return sigma: a (batch_size, n) shaped tensor, containing the forecasted standard devations
"""
def get_mu_sigma(y_pred, n):
    mu = tf.gather(y_pred, range(n), axis=1)
    sigma = tf.gather(y_pred, range(n, 2*n), axis=1)
    sigma = activations_stdev(sigma)
    return mu, sigma


"""
Computes CRPS. TODO: cite the source of this CRPS expression

:param mu: a (batch_size, n) shaped tensor, containing the forecasted means
:param sigma: a (batch_size, n) shaped tensor, containing the forecasted standard deviations
:param y_true: a (batch_size, n) shaped tensor, containing the observations

:return: a (1,) shaped tensor, containing the mean CRPS
"""
def gaussian_CRPS(mu, sigma, y_true):
    dist = tfd.Normal(loc=0., scale=1.)
    y_0 = (y_true - mu)/sigma
    CRPS = sigma * (y_0 * (2 * dist.cdf(y_0)- 1) + 2 * dist.prob(y_0) - 1/tfm.sqrt(pi))
    return tf.reduce_mean(CRPS)


"""
Implementation of a keras custom loss function. It evaluates CRPS over a Gaussian generated CRPS term.

The source for this formula is given by:
    Gneiting, T, A E Raftery, A H Westveld III, and T Goldman.
    2005. “Calibrated Probabilistic Forecasting Using Ensemble
    Model Output Statistics and Minimum CRPS Estimation.” Monthly
    Weather Review 133: 1098–1118.

:param y_true: a (batch_size, n) shaped tensor, containing the measurements
:param y_pred: a (batch_size, 2*n) shaped tensor, with:
                    y_pred[:,:n] containing the forecasted means
                    y_pred[:,n:] containing the forecasted standard deviations
:return: a (1,) shaped tensor, the mean CRPS computed over the forecasted distributions
"""
def gaussian_CRPS_loss(y_true, y_pred):
    n = y_true.shape[1]
    mu, sigma = get_mu_sigma(y_pred, n)
    return gaussian_CRPS(mu, sigma, y_true)
    