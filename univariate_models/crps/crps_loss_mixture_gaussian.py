import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpm = tfp.math

# Allow for the parentfolder to be imported
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from univariate_models.mixture_gaussian_helpers import repeat_var, create_mesh, preprocess_mixture_output


"""
    Helper function, in accordance with http://cran.nexr.com/web/packages/scoringRules/vignettes/crpsformulas.html
"""
def A(mu, sigma):
    dist = tfd.Normal(loc=0., scale=1.)
    return 2 * sigma * dist.prob(mu/sigma) + mu * (2 * dist.cdf(mu/sigma)-1)



"""
    Implements CRPS loss for a mixture Gaussian distribution. In accordance with http://cran.nexr.com/web/packages/scoringRules/vignettes/crpsformulas.html
    
    The source for this formula is:
        Grimit, E P, T Gneiting, V J Berrocal, and N A Johnson. 2006.
        “The Continuous Ranked Probability Score for Circular Variables
        and Its Application to Mesoscale Forecast Ensemble Verification.”
        Quarterly Journal of the Royal Meteorological Society 132: 2925–42.
    
    :param y_true: a (batch_size, n_targets) shaped tensor
    :param means: a (batch_size, n_mixtures, n) shaped tensor containing each mixture's mean
    :param stdevs: a (batch_size, n_mixtures, d) shaped tensor containing each mixture's standard deviation
    :param weights: a (batch_size, n_mixtures, n_targets) shaped tensor, containing each mixture's weight.
"""
def mixture_gaussian_CRPS(y_true, means, stdevs, weights):
    n_mixtures = means.shape[1]
    
    # First transpose the last two dimensions, we might want to remove this transposing later on
    # shapes are now (batch_size, d, n_mixtures)
    means = tf.transpose(means, perm=[0, 2, 1])
    stdevs = tf.transpose(stdevs, perm=[0, 2, 1])
    weights = tf.transpose(weights, perm=[0, 2, 1])
    
    # Create a (batch_size, d, n_mixtures) shaped observation variable
    true2 = repeat_var(y_true, n_mixtures)
    
    # Create two tensors of (batch_size, d, n_mixtures, n_mixtures)
    #Any operation on these two arrays is an operation on possible combinations of mixtures
    weightsArr1, weightsArr2 = create_mesh(weights)
    meansArr1, meansArr2 = create_mesh(means)
    stdevArr1, stdevArr2 = create_mesh(stdevs)
    
    # Term 1, in accordance with the website source
    CRPS1 = weights * A(true2 - means, stdevs)
    CRPS1 = tf.reduce_sum(CRPS1, axis=2)

    # Term 2, in accordance with the website source
    CRPS2 = 0.5 * weightsArr1 * weightsArr2 * A(meansArr1 - meansArr2, tf.math.sqrt(stdevArr1**2 + stdevArr2**2))

    # Reduce all dimensions except for the batch dimension (this could have been done in a single step)
    CRPS2 = tf.reduce_sum(CRPS2, axis=3)
    CRPS2 = tf.reduce_sum(CRPS2, axis=2)

    #Return the final CRPS expression
    return tf.reduce_mean(CRPS1 - CRPS2)



"""
    Wrapper to use mixture Gaussian CRPS as a keras loss function
    
    :param y_true: an (batch_size, n_targets) shaped tensor
    :param y_pred: an (batch_size, n_mixtures, 3*n_targets) shaped tensor
    
    :return the mean mixture Gaussian CRPS
"""
def mixture_gaussian_CRPS_loss(y_true, y_pred):
    n_targets = y_true.shape[1]
    means, stdevs, weights = preprocess_mixture_output(y_pred, n_targets)

    return mixture_gaussian_CRPS(y_true, means, stdevs, weights)
