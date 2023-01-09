import tensorflow as tf
import keras
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpm = tfp.math


"""
    Activation of standard deviations, mainly to prevent zero division
"""
def activation_stdevs(stdevs):
    return 0.001+keras.activations.softplus(stdevs)
    
"""
    Activation/normalization for the predicted weights.
    
    :param weights: an (batch_size, d, n_mixtures) shaped array.
    :param n_mixtures: an integer
    :return weights: an (batch_size, d, n_mixtures) shaped array.
    
    for all values of batch_size and d, we have: sum(i=1 to n_mixtures) weights[batch_size, d, i] = 1
"""
def activation_weights(weights, n_mixtures):
    weights = 0.001+0.999*keras.activations.sigmoid(weights)
    sums = tf.reduce_sum(weights, axis=1)
    sums = repeat_var(sums, n_mixtures, axis=1)
    return weights / sums


"""
    Creates two tensors containing all possible values.
    
    :param inp_arr: an (batch_size, d, n_mixtures) sized tensor
    :return arr: an (batch_size, d, n_mixtures, n_mixtures) sized tensor
    :return arr2: an (batch_size, d, n_mixtures, n_mixtures) sized tensor
    
    If inp_arr[a,b,c] = val, then for all 1 <= i <= d:
        arr[a,b,i,d] = val
        arr2[a, b, d, i] = val
"""
def create_mesh(inp_arr):
    arr = tf.expand_dims(inp_arr, axis=2)
    arr = tf.repeat(arr, repeats=arr.shape[3], axis=2)
    arr2 = tf.transpose(arr, perm = [0, 1, 3, 2])
    return arr, arr2



"""
    Creates a tensor with repeated values
    
    :param y_true: an (batch_size, d) sized tensor
    :param n_mixtures: an integer
    :param true2: an (batch_size, d, n_mixtures) sized tensor
"""
def repeat_var(y_true, n_repeats, axis = 2):
    true2 = tf.expand_dims(y_true, axis)
    true2 = tf.repeat(true2, repeats=n_repeats, axis=axis)
    return true2



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
    
    :param y_true: a (batch_size, d) shaped tensor
    :param means: a (batch_size, n_mixtures, d) shaped tensor containing each mixture's mean
    :param stdevs: a (batch_size, n_mixtures, d) shaped tensor containing each mixture's standard deviation
    :param weights: a (batch_size, n_mixtures, d) shaped tensor, containing each mixture's weight.
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
    Cuts the (batch_size, n_mixtures, 3*d) model output in three (activated) tensors of shape (batch_size, n_mixtures, d)
"""
def preprocess_mixture_output(y_pred, d):
    n_mixtures = y_pred.shape[1]
    # Cut the predictions into three parts, denoting the means, standard deviations and weights of the mixtures
    means = tf.gather(y_pred, range(d), axis=2)
    stdevs = tf.gather(y_pred, range(d, 2*d), axis=2)
    weights = tf.gather(y_pred, range(2*d, 3*d), axis=2)

    # Activate them to cast them to the required intervals
    stdevs = activation_stdevs(stdevs)
    weights = activation_weights(weights, n_mixtures)
    
    return means, stdevs, weights


"""
    Wrapper to use mixture Gaussian CRPS as a keras loss function
    
    :param y_true: an (batch_size, d) shaped tensor
    :param y_pred: an (batch_size, n_mixtures, 3*d) shaped tensor
    
    :return the mean mixture Gaussian CRPS
"""
def mixture_gaussian_CRPS_loss(y_true, y_pred):
    d = y_true.shape[1]
    means, stdevs, weights = preprocess_mixture_output(y_pred, d)

    return mixture_gaussian_CRPS(y_true, means, stdevs, weights)