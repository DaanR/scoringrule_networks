import tensorflow as tf
import keras

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
def get_mu_sigma(y_pred, n_targets):
    mu = tf.gather(y_pred, range(n_targets), axis=1)
    sigma = tf.gather(y_pred, range(n_targets, 2*n_targets), axis=1)
    sigma = activations_stdev(sigma)
    return mu, sigma
