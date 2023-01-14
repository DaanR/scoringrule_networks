import tensorflow as tf
import keras



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
    Cuts the (batch_size, n_mixtures, 3*n_targets) model output in three (activated) tensors of shape (batch_size, n_mixtures, n_targets)
"""
def preprocess_mixture_output(y_pred, n_targets):
    n_mixtures = y_pred.shape[1]
    # Cut the predictions into three parts, denoting the means, standard deviations and weights of the mixtures
    means = tf.gather(y_pred, range(n_targets), axis=2)
    stdevs = tf.gather(y_pred, range(n_targets, 2*n_targets), axis=2)
    weights = tf.gather(y_pred, range(2*n_targets, 3*n_targets), axis=2)

    # Activate them to cast them to the required intervals
    stdevs = activation_stdevs(stdevs)
    weights = activation_weights(weights, n_mixtures)
    
    return means, stdevs, weights